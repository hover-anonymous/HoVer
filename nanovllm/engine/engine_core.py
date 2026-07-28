import time
import os
import logging
import threading
import torch.multiprocessing as mp
import queue
import zmq
import pickle
from contextlib import ExitStack

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.config import Config
from nanovllm.request_type import EngineCoreRequestType
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.output_protocol import (
    build_engine_output_batch,
)
from nanovllm.utils.zmq_utils import make_zmq_socket
from nanovllm.utils.utils import disable_gc


logger = logging.getLogger("engine_core")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(process)d - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)




class EngineCore:
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.config = config
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        if config.tensor_parallel_size > 1:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
                
        self.model_runner = ModelRunner(config, 0, self.events if config.tensor_parallel_size > 1 else [])
        logger.info("Engine core model runner loaded")
        self.scheduler = Scheduler(config)
        self.scheduler.model_num_layers = int(
            getattr(self.model_runner, "num_layers", 0) or 0
        )
        
        # HoVer policies that consume live expert/cache state need the
        # rank-0 runner.  HoVer uses it for C2 interference admission and
        # predictor-aware ordering; without this reference those features
        # silently fall back to their disabled paths.
        self.scheduler.model_runner = self.model_runner
        logger.info(
            "HoVer scheduling enabled, model_runner reference set in scheduler",
        )

        
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[Sequence]()
        self.input_address = input_address
        self.output_address = output_address
        logger.info(f"Engine core input address: {self.input_address}")
        logger.info(f"Engine core output address: {self.output_address}")
        self.input_thread = threading.Thread(target=self.process_input_sockets,
                         args=(self.input_address,),
                         daemon=True)
        self.input_thread.start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            args=(self.output_address,),
            daemon=True)
        self.output_thread.start()

    def exit(self):
        pass
        # self._send_engine_dead()
        # self.model_runner.call("exit")
        # del self.model_runner
        # for p in self.ps:
        #     p.join()
        # logger.info("Engine core model runner exit")

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        # Reload hf_config in the spawned process because AutoConfig may not be pickle-safe.
        if config.hf_config is None:
            try:
                from transformers import AutoConfig
                import torch
                logger.info("Reloading AutoConfig in EngineCore process...")

                
                trust_remote = getattr(config, 'trust_remote_code', False)
                if trust_remote:
                    import sys
                    from pathlib import Path

                    source = os.environ.get("DEEPSEEK_VL2_PATH")
                    if source:
                        path = Path(source).expanduser().resolve()
                        if not path.is_dir():
                            raise ValueError(
                                f"DEEPSEEK_VL2_PATH is not a directory: {path}"
                            )
                        if str(path) not in sys.path:
                            sys.path.insert(0, str(path))
                    try:
                        import deepseek_vl2.models  # noqa: F401
                    except ImportError:
                        pass

                config.hf_config = AutoConfig.from_pretrained(
                    config.model, trust_remote_code=trust_remote
                )
                
                if not hasattr(config.hf_config, "max_position_embeddings"):
                    lang_cfg = getattr(config.hf_config, "language_config", None)
                    if lang_cfg is not None:
                        max_pos = lang_cfg.get("max_position_embeddings") if isinstance(lang_cfg, dict) \
                                  else getattr(lang_cfg, "max_position_embeddings", None)
                        if max_pos is not None:
                            config.hf_config.max_position_embeddings = max_pos

                if not getattr(config.hf_config, "torch_dtype", None):
                    config.hf_config.torch_dtype = torch.bfloat16
                # max_model_len is already set in parent, no need to re-min()
            except Exception as e:
                logger.error(f"Failed to reload AutoConfig: {e}")
                raise

        engine : EngineCore = None
        try:
            engine = EngineCore(config, input_address, output_address)
            engine.busy_loop()
        except Exception as e:
            logger.error(f"Engine core exception: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
        finally:
            if engine is not None:
                engine.exit()

    def _send_engine_dead(self):
        logger.info("Send engine core dead signal")
        exit_seq = Sequence([-1], seq_id=-1)
        self.output_queue.put_nowait([exit_seq])
        self.output_thread.join(timeout=5.0)

    def busy_loop(self):
        shutdown = False
        while True:
            s = time.time()
            shutdown = shutdown or self._process_input_queue()
            m = time.time()
            self._process_engine_step()
            e = time.time()
            duration = e - s
            if duration > 0.08:
                logger.warning(f"Engine core step took too long: {duration:.3f}s")
                print(f"  Input queue processing time: {m - s:.3f}s"
                      f"  Engine step time: {e - m:.3f}s")
            if shutdown:
                break

    def _process_input_queue(self):
        while not self.input_queue.empty():
            seq = self.input_queue.get_nowait()
            if seq.seq_id == -1:
                logger.info("Engine core input thread shutdown")
                return True
            try:
                self.scheduler.add(seq)
            except Exception as e:
                logger.error(f"Failed to add sequence {seq.seq_id} to scheduler: {e}")
                seq.set_error(str(e))
                self.output_queue.put_nowait([seq])
        return False


    @disable_gc()
    def _process_engine_step(self):
        schedule_start_ns = time.perf_counter_ns()
        seqs = self.scheduler.schedule()
        self.scheduler.batch_schedule_ns = (
            time.perf_counter_ns() - schedule_start_ns
        )
        self.scheduler.batch_schedule_ms = (
            self.scheduler.batch_schedule_ns / 1e6
        )
        if not seqs:
            return

        PREF = SequenceStatus.PREFILLING
        DEC = SequenceStatus.DECODING
        FIN = SequenceStatus.FINISHED
        pref_seqs = [seq for seq in seqs if seq.status == PREF]
        decode_count = sum(seq.status == DEC for seq in seqs)
        prefill_tokens = sum(
            int(getattr(seq, "num_tokens_to_process", 0))
            for seq in pref_seqs
        )
        active_stage = -1
        active_layers = 0
        prefill_layer_work = 0
        active_num_stages = -1
        model_layers = int(
            getattr(self.model_runner, "num_layers", 0) or 0
        )
        if pref_seqs:
            first_prefill = pref_seqs[0]
            active_stage = int(getattr(first_prefill, "stage", -1))
            active_num_stages = int(
                getattr(first_prefill, "num_stages", -1)
            )
            if (
                active_stage >= 0
                and active_num_stages > 0
                and model_layers > 0
            ):
                quotient, remainder = divmod(
                    model_layers, active_num_stages
                )
                active_layers = quotient + int(active_stage < remainder)
            else:
                active_layers = model_layers
            prefill_layer_work = sum(
                int(getattr(seq, "num_tokens_to_process", 0))
                * active_layers
                for seq in pref_seqs
            )

        service_start = time.perf_counter()
        token_ids = self.model_runner.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids)
        finished_seqids = [
            seq.seq_id for seq in seqs if seq.status == FIN
        ]
        if finished_seqids:
            self.model_runner.call(
                "release_multimodal_cache", finished_seqids
            )
        service_ms = (time.perf_counter() - service_start) * 1000.0
        self.scheduler.observe_round_service(
            service_ms,
            prefill_layer_work,
            decode_count,
            model_layers,
            prefill_tokens=prefill_tokens,
            active_layers=active_layers,
            active_stage=active_stage,
            num_stages=active_num_stages,
        )
        self.output_queue.put_nowait(seqs)


    def process_input_sockets(self, input_address: str):
        """Input socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(make_zmq_socket(ctx,
                                    input_address,
                                    zmq.DEALER,
                                    bind=False))
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
            input_socket.send(b'')
            poller.register(input_socket, zmq.POLLIN)
            logger.info("Engine core input socket connected")

            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    serialized_obj = input_socket.recv(copy=False)
                    obj = pickle.loads(serialized_obj)
                    request_type = obj[0]
                    if (request_type == EngineCoreRequestType.ADD):
                        self.input_queue.put_nowait(obj[1])
                    elif (request_type == EngineCoreRequestType.SHUTDOWN):
                        logger.info("Engine core input thread shutdown")
                        self.input_queue.put_nowait(Sequence([-1], seq_id=-1))
                        break

    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        wire_mode = os.environ.get(
            "NANOVLLM_OUTPUT_WIRE_MODE", "legacy"
        ).strip().lower()
        if wire_mode not in {"dto", "legacy"}:
            raise RuntimeError(
                "NANOVLLM_OUTPUT_WIRE_MODE must be 'dto' or 'legacy'; "
                f"got {wire_mode!r}"
            )
        logger.info("Engine output wire mode: %s", wire_mode)
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000))
            logger.info("Engine core output socket connected")

            while True:
                output = self.output_queue.get()
                if any(seq.seq_id == -1 for seq in output):
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None), protocol=pickle.HIGHEST_PROTOCOL))
                    logger.info("Engine core output thread closed")
                    return

                if wire_mode == "dto":
                    wire_payload = build_engine_output_batch(output)
                else:
                    wire_payload = output
                serialized_obj = pickle.dumps(
                    (EngineCoreRequestType.ADD, wire_payload),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                socket.send(serialized_obj)
