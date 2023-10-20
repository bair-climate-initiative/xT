from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.cli import SaveConfigCallback
from pytorch_lightning.loggers import Logger


class PreEvalCallBack(Callback):
    def on_train_start(self, trainer, pl_module):
        return trainer.validate()


class LoggerSaveConfigCallback(SaveConfigCallback):
    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if self.already_saved:
            return
        super().setup(trainer, pl_module, stage)
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(
                self.config, skip_none=False
            )  # Required for proper reproducibility
            trainer.logger.log_hyperparams(self.config)
