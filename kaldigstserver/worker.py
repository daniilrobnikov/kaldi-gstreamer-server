__author__ = 'tanel'

import logging
import logging.config
from subprocess import Popen, PIPE
from gi.repository import GLib
import argparse
import yaml
import json
import sys
import zlib
import base64
import time

import asyncio
import tornado.process
import tornado.ioloop
import tornado.websocket
from concurrent.futures import ThreadPoolExecutor

from decoder import DecoderPipeline
from decoder2 import DecoderPipeline2

import common


logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=5)

CONNECT_TIMEOUT = 5
SILENCE_TIMEOUT = 5
USE_NNET2 = False


def get_args():
    parser = argparse.ArgumentParser(
        description='Worker for next-gen-kaldi-server')

    parser.add_argument(
        '-u',
        '--uri',
        default="ws://localhost:8888/worker/ws/speech",
        dest="uri",
        help="URI of the server (default: ws://localhost:8888/worker/ws/speech)",
    )

    parser.add_argument(
        '-f',
        '--fork',
        default=1,
        dest="fork",
        type=int,
        help="Number of processes to fork",
    )

    parser.add_argument(
        '-c',
        '--conf',
        dest="conf",
        help="YAML file with decoder configuration",
    )

    return parser.parse_args()


class Worker():
    STATE_CREATED = 0
    STATE_CONNECTED = 1
    STATE_INITIALIZED = 2
    STATE_PROCESSING = 3
    STATE_EOS_RECEIVED = 7
    STATE_CANCELLING = 8
    STATE_FINISHED = 100

    def __init__(self, uri, decoder_pipeline, post_processor, full_post_processor=None):
        self.uri = uri
        self.decoder_pipeline = decoder_pipeline
        self.post_processor = post_processor
        self.full_post_processor = full_post_processor
        self.pipeline_initialized = False
        self.partial_transcript = ""
        if USE_NNET2:
            self.decoder_pipeline.set_result_handler(self._on_result)
            self.decoder_pipeline.set_full_result_handler(self._on_full_result)
        else:
            self.decoder_pipeline.set_word_handler(self._on_word)

        self.decoder_pipeline.set_error_handler(self._on_error)
        self.decoder_pipeline.set_eos_handler(self._on_eos)
        self.state = self.STATE_CREATED
        self.request_id = "<undefined>"
        self.timeout_decoder = 5
        logger.error(f"Using decoder timeout {self.timeout_decoder}")
        self.num_segments = 0
        self.last_partial_result = ""
        self.post_processor_lock = asyncio.Lock()
        self.processing_condition = asyncio.Condition()
        self.num_processing_threads = 0

    async def connect_and_run(self):
        logger.info("Opening websocket connection to master server")
        self.ws = await tornado.websocket.websocket_connect(self.uri, ping_interval=10)
        logger.info("Opened websocket connection to server")
        self.state = self.STATE_CONNECTED
        self.last_partial_result = ""
        self.last_decoder_message = time.time()
        logger.error(f"Starting decoder run {self.last_decoder_message}")
        while True:
            msg = await self.ws.read_message()
            await self.received_message(msg)
            if msg is None:
                await self.closed()
                break
        logger.info("Finished decoding run")

    async def guard_timeout(self):
        global SILENCE_TIMEOUT
        while self.state in [self.STATE_EOS_RECEIVED, self.STATE_CONNECTED, self.STATE_INITIALIZED, self.STATE_PROCESSING]:
            if time.time() - self.last_decoder_message > SILENCE_TIMEOUT:
                logger.warning(
                    f"{self.request_id}: More than {self.SILENCE_TIMEOUT} seconds from last decoder hypothesis update, cancelling"
                )
                await self.finish_request()

                event = dict(status=common.STATUS_NO_SPEECH)
                try:
                    await self.ws.write_message(json.dumps(event))
                except:
                    logger.warning(
                        f"{self.request_id}: Failed to send error event to master"
                    )

                await self.ws.close()
                return
            logger.debug(
                f"{self.request_id}: Checking that decoder hasn't been silent for more than {self.SILENCE_TIMEOUT} seconds"
            )
            await asyncio.sleep(1)

    async def received_message(self, m):
        logger.debug(
            f"{self.request_id}: Got message from server of type {type(m)}")
        if self.state == self.STATE_CONNECTED:
            props = json.loads(m)
            content_type = props['content_type']
            self.request_id = props['id']
            self.num_segments = 0
            # self.decoder_pipeline.init_request(self.request_id, content_type)
            logger.error(
                f"Initialized decoder {self.decoder_pipeline.init_request(self.request_id, content_type)}")
            self.last_decoder_message = time.time()
            logger.error(f"Starting timeout guard {self.last_decoder_message}")
            # tornado.ioloop.IOLoop.current().run_in_executor(executor, self.guard_timeout)
            asyncio.create_task(self.guard_timeout())
            logger.info(f"{self.request_id}: Started timeout guard")
            logger.info(f"{self.request_id}: Initialized request")
            self.state = self.STATE_INITIALIZED
        elif m == "EOS":
            if self.state not in [self.STATE_CANCELLING, self.STATE_EOS_RECEIVED, self.STATE_FINISHED]:
                # self.decoder_pipeline.end_request()
                logger.info(
                    f"Sent EOS to decoder {self.decoder_pipeline.end_request()}")
                self.state = self.STATE_EOS_RECEIVED
            else:
                logger.info(
                    f"{self.request_id}: Ignoring EOS, worker already in state {self.state}")
        else:
            if self.state not in [self.STATE_CANCELLING, self.STATE_EOS_RECEIVED, self.STATE_FINISHED]:
                if isinstance(m, bytes):
                    # self.decoder_pipeline.process_data(m)
                    logger.error(
                        f"Sent data to decoder {self.decoder_pipeline.process_data(m)}")
                    self.state = self.STATE_PROCESSING
                elif isinstance(m, str):
                    props = json.loads(str(m))
                    if 'adaptation_state' in props:
                        as_props = props['adaptation_state']
                        if as_props.get('type', "") == "string+gzip+base64":
                            adaptation_state = zlib.decompress(base64.b64decode(
                                as_props.get('value', ''))).decode("utf-8")
                            logger.info(
                                f"{self.request_id}: Setting adaptation state to user-provided value")
                            # self.decoder_pipeline.set_adaptation_state(adaptation_state)
                            logger.error(
                                f"Sent adaptation state to decoder {self.decoder_pipeline.set_adaptation_state(adaptation_state)}")
                        else:
                            logger.warning(
                                f"{self.request_id}: Cannot handle adaptation state type {as_props.get('type', '')}")
                    else:
                        logger.warning(
                            f"{self.request_id}: Got JSON message but don't know what to do with it")
            else:
                logger.info(
                    f"{self.request_id}: Ignoring data, worker already in state {self.state}")

    async def finish_request(self):
        if self.state in (self.STATE_CONNECTED, self.STATE_INITIALIZED):
            # connection closed when we are not doing anything
            # or request initialized but with no data sent
            # self.decoder_pipeline.finish_request()
            logger.error(
                f"Finished request {self.decoder_pipeline.finish_request()}")
            self.state = self.STATE_FINISHED
            return

        if self.state != self.STATE_FINISHED:
            logger.info(
                f"{self.request_id}: Master disconnected before decoder reached EOS?")
            self.state = self.STATE_CANCELLING
            # self.decoder_pipeline.cancel()
            logger.error(f"Cancelled decoder {self.decoder_pipeline.cancel()}")

            timeout_seconds = 30

            try:
                await asyncio.wait_for(self._wait_for_decoder_eos(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.info(
                    f"{self.request_id}: Giving up waiting after {timeout_seconds} seconds")
                self.state = self.STATE_FINISHED

            # self.decoder_pipeline.finish_request()
            logger.error(
                f"Finished request {self.decoder_pipeline.finish_request()}")
            logger.info(f"{self.request_id}: Finished waiting for EOS")

    async def _wait_for_decoder_eos(self):
        while self.state == self.STATE_CANCELLING:
            logger.info(f"{self.request_id}: Waiting for EOS from decoder")
            await asyncio.sleep(1)

    async def closed(self):
        logger.debug(f"{self.request_id}: Websocket closed() called")
        await self.finish_request()
        logger.debug(f"{self.request_id}: Websocket closed() finished")

    async def _increment_num_processing(self, delta):
        async with self.processing_condition:
            self.num_processing_threads += delta
            self.processing_condition.notify()

    async def _on_result(self, result, final):
        try:
            await self._increment_num_processing(1)

            if final:
                # final results are handled by _on_full_result()
                return
            self.last_decoder_message = time.time()
            logger.error(f"Updated timeout guard {self.last_decoder_message}")
            if self.last_partial_result == result:
                return
            self.last_partial_result = result
            logger.info(
                f"{self.request_id}: Postprocessing (final={final}) result..")
            processed_transcripts = await self.post_process([result], blocking=False)
            if processed_transcripts:
                logger.info(f"{self.request_id}: Postprocessing done.")
                event = dict(status=common.STATUS_SUCCESS,
                             segment=self.num_segments,
                             result=dict(hypotheses=[dict(transcript=processed_transcripts[0])], final=final))
                try:
                    self.ws.write_message(json.dumps(event))
                except:
                    e = sys.exc_info()[1]
                    logger.warning(f"Failed to send event to master: {e}")
        finally:
            await self._increment_num_processing(-1)

    async def _on_full_result(self, full_result_json):
        try:
            await self._increment_num_processing(1)

            self.last_decoder_message = time.time()
            logger.error(f"Updated timeout guard {self.last_decoder_message}")
            full_result = json.loads(full_result_json)
            full_result['segment'] = self.num_segments
            full_result['id'] = self.request_id
            if full_result.get("status", -1) == common.STATUS_SUCCESS:
                logger.debug(
                    f"{self.request_id}: Before postprocessing: {repr(full_result)}")
                full_result = await self.post_process_full(full_result)
                logger.info("%s: Postprocessing done." % self.request_id)
                logger.debug(u"%s: After postprocessing: %s" %
                             (self.request_id, repr(full_result)))

                try:
                    await self.ws.write_message(json.dumps(full_result))
                except:
                    e = sys.exc_info()[1]
                    logger.warning("Failed to send event to master: %s" % e)
                if full_result.get("result", {}).get("final", True):
                    self.num_segments += 1
                    self.last_partial_result = ""
            else:
                logger.info("%s: Result status is %d, forwarding the result to the server anyway" % (
                    self.request_id, full_result.get("status", -1)))
                try:
                    await self.ws.write_message(json.dumps(full_result))
                except:
                    e = sys.exc_info()[1]
                    logger.warning("Failed to send event to master: %s" % e)
        finally:
            await self._increment_num_processing(-1)

    async def _on_word(self, word):
        try:
            await self._increment_num_processing(1)

            self.last_decoder_message = time.time()
            logger.error(f"Current word {word}")
            logger.error(
                f"Current partial transcript {[self.partial_transcript]}")
            if word != "<#s>":
                if len(self.partial_transcript) > 0:
                    self.partial_transcript += " "
                self.partial_transcript += word
                logger.debug("%s: Postprocessing partial result.." %
                             self.request_id)
                processed_transcripts = await self.post_process([self.partial_transcript], blocking=False)
                if processed_transcripts:
                    logger.debug(f"{self.request_id}: Postprocessing done.")

                    event = dict(status=common.STATUS_SUCCESS,
                                 segment=self.num_segments,
                                 result=dict(hypotheses=[dict(transcript=processed_transcripts[0])], final=False))
                    await self.ws.write_message(json.dumps(event))
            else:
                logger.info("%s: Postprocessing final result.." %
                            self.request_id)
                processed_transcripts = await self.post_process([self.partial_transcript], blocking=True)
                logger.info("%s: Postprocessing done." % self.request_id)
                event = dict(status=common.STATUS_SUCCESS,
                             segment=self.num_segments,
                             result=dict(hypotheses=[dict(transcript=processed_transcripts[0])], final=True))
                await self.ws.write_message(json.dumps(event))
                self.partial_transcript = ""
                self.num_segments += 1
        finally:
            await self._increment_num_processing(-1)

    async def _on_eos(self, data=None):
        self.last_decoder_message = time.time()
        logger.error(f"Updated timeout guard {self.last_decoder_message}")
        # Make sure we won't close the connection before the
        # post-processing has finished
        while self.num_processing_threads > 0:
            logging.debug(
                f"Waiting until processing threads finish ({self.num_processing_threads})")
            await self.processing_condition.wait()

        self.state = self.STATE_FINISHED
        await self.send_adaptation_state()

        if hasattr(self.ws, 'close') and asyncio.iscoroutinefunction(self.ws.close):
            await self.ws.close()
        else:
            self.ws.close()

    async def _on_error(self, error_message):
        self.state = self.STATE_FINISHED
        event = {
            "status": common.STATUS_NOT_ALLOWED,
            "message": error_message,
        }
        try:
            await self.ws.write_message(json.dumps(event))
        except Exception as e:
            logger.warning(f"Failed to send event to master: {e}")
        await self.ws.close()

    async def send_adaptation_state(self):
        if hasattr(self.decoder_pipeline, 'get_adaptation_state'):
            logger.info("%s: Sending adaptation state to client..." %
                        (self.request_id))
            adaptation_state = self.decoder_pipeline.get_adaptation_state()
            logger.error(f"Adaptation state: {adaptation_state}")
            event = dict(status=common.STATUS_SUCCESS,
                         adaptation_state=dict(id=self.request_id,
                                               value=base64.b64encode(zlib.compress(
                                                   adaptation_state.encode())).decode("utf-8"),
                                               type="string+gzip+base64",
                                               time=time.strftime("%Y-%m-%dT%H:%M:%S")))
            try:
                await self.ws.write_message(json.dumps(event))
            except Exception as e:
                logger.warning(f"Failed to send event to master: {e}")
        else:
            logger.info(
                f"{self.request_id}: Adaptation state not supported by the decoder, not sending it.")

    async def post_process(self, texts, blocking=False):
        if self.post_processor:
            logging.debug(
                f"{self.request_id}: Waiting for postprocessor lock with blocking={blocking}")
            timeout = None if blocking else 0.1

            try:
                acquired = await asyncio.wait_for(self.post_processor_lock.acquire(), timeout=timeout)
                if acquired:
                    try:
                        result = []
                        for text in texts:
                            try:
                                logging.debug(
                                    f"{self.request_id}: Starting postprocessing: {text}")
                                self.post_processor.stdin.write(
                                    (text + "\n").encode("utf-8"))
                                self.post_processor.stdin.flush()
                                logging.debug(
                                    f"{self.request_id}: Reading from postpocessor")
                                text = await self.post_processor.stdout.read_until(b'\n')
                                text = text.decode("utf-8").strip()
                                logging.debug(
                                    f"{self.request_id}: Postprocessing returned: {text}")
                                text = text.replace("\\n", "\n")
                                result.append(text)
                            except Exception as ex:
                                logging.exception("Error when postprocessing")
                    finally:
                        self.post_processor_lock.release()
                    return result
            except asyncio.TimeoutError:
                logging.info(
                    f"{self.request_id}: Skipping postprocessing since post-processor already in use")
                return None
        else:
            return texts

    async def post_process_full(self, full_result):
        if self.full_post_processor:
            self.full_post_processor.stdin.write(
                "%s\n\n" % json.dumps(full_result))
            self.full_post_processor.stdin.flush()
            lines = []
            while True:
                l = await self.full_post_processor.stdout.readline()
                if not l:
                    break  # EOF
                if l.strip() == "":
                    break
                lines.append(l)
            full_result = json.loads("".join(lines))

        elif self.post_processor:
            transcripts = []
            for hyp in full_result.get("result", {}).get("hypotheses", []):
                transcripts.append(hyp["transcript"])
            processed_transcripts = await self.post_process(transcripts, blocking=True)
            for (i, hyp) in enumerate(full_result.get("result", {}).get("hypotheses", [])):
                hyp["original-transcript"] = hyp["transcript"]
                hyp["transcript"] = processed_transcripts[i]
        return full_result


async def main_loop(uri, decoder_pipeline, post_processor, full_post_processor=None):
    while True:
        worker = Worker(uri, decoder_pipeline, post_processor,
                        full_post_processor=full_post_processor)
        try:
            await worker.connect_and_run()
        except Exception:
            logger.error(
                "Couldn't connect to server, waiting for %d seconds", CONNECT_TIMEOUT)
            await asyncio.sleep(CONNECT_TIMEOUT)
        # fixes a race condition
        await asyncio.sleep(1)


def load_configuration(args):
    """
    Load configuration from YAML file
    """
    conf = {}
    if args.conf:
        with open(args.conf) as f:
            conf = yaml.safe_load(f)
    return conf


def initialize_post_processors(conf):
    """
    Fork off the post-processors before load the model into memory
    """
    tornado.process.Subprocess.initialize()
    post_processor = None
    if "post-processor" in conf:
        STREAM = tornado.process.Subprocess.STREAM
        post_processor = tornado.process.Subprocess(
            conf["post-processor"], shell=True, stdin=PIPE, stdout=STREAM, )

    full_post_processor = None
    if "full-post-processor" in conf:
        full_post_processor = Popen(
            conf["full-post-processor"], shell=True, stdin=PIPE, stdout=PIPE)
    return post_processor, full_post_processor


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)8s %(asctime)s %(message)s ")
    logging.debug('Starting up worker')

    args = get_args()

    if args.fork > 1:
        logging.info(f"Forking into {args.fork} processes")
        tornado.process.fork_processes(args.fork)

    conf = load_configuration(args)

    if "logging" in conf:
        logging.config.dictConfig(conf["logging"])

    post_processor, full_post_processor = initialize_post_processors(conf)

    global USE_NNET2
    USE_NNET2 = conf.get("use-nnet2", False)

    global SILENCE_TIMEOUT
    SILENCE_TIMEOUT = conf.get("silence-timeout", 5)
    if USE_NNET2:
        decoder_pipeline = DecoderPipeline2(
            tornado.ioloop.IOLoop.current(), conf)
    else:
        decoder_pipeline = DecoderPipeline(
            tornado.ioloop.IOLoop.current(), conf)

    glib_loop = GLib.MainLoop()
    tornado.ioloop.IOLoop.current().run_in_executor(executor, glib_loop.run)
    tornado.ioloop.IOLoop.current().spawn_callback(
        main_loop, args.uri, decoder_pipeline, post_processor, full_post_processor)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
