import logging
import traceback

ENV_LOCAL = "local"
ENV_DEV = "dev"
ENV_PROD = "prod"

class Logger:

    def __init__(self, name, env):
        self.env = env
        self.logger: logging.Logger = self._init_logging(env, name)

    def _init_logging(self, env, name):
        if env in [ENV_LOCAL, ENV_DEV]:
            level = logging.DEBUG
            handlers = [logging.StreamHandler()]
        elif env == ENV_PROD:
            level = logging.INFO
            handlers = [
                logging.FileHandler("flmpid.log", encoding="utf-8"),
            ]

        logging.basicConfig(
            format="%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=level,
            handlers=handlers,
        )

        if env in [ENV_LOCAL, ENV_DEV]:
            logging.getLogger("net").setLevel(logging.WARNING)
            logging.getLogger("scheduler").setLevel(logging.WARNING)
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
            logging.getLogger("PIL").setLevel(logging.WARNING)
            logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
        elif env == ENV_PROD:
            logging.getLogger("net").setLevel(logging.INFO)
            logging.getLogger("scheduler").setLevel(logging.INFO)
            logging.getLogger('matplotlib').setLevel(logging.INFO)
            logging.getLogger("apscheduler.scheduler").setLevel(logging.INFO)
            logging.getLogger("PIL").setLevel(logging.WARNING)
            logging.getLogger("apscheduler.executors.default").setLevel(logging.INFO)

        return logging.getLogger(name)

    def set_level(self, level):
        self.logger.setLevel(level)

    def log_exception(self, e: Exception):
        error_trace = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        self.logger.error(error_trace)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not self.is_debug():
            return
        self.logger.warning(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if not self.is_debug():
            return
        self.logger.debug(msg, *args, **kwargs)

    def is_debug(self):
        return self.env in [ENV_LOCAL, ENV_DEV]
