from seqsketch.utils.config import Config
file = "test.yaml"

configurator = Config(
        config_file="seqsketch/configs/" + file )
config = configurator.get_config()
dataloader = configurator.get_dataloader()
print("Init model")
model = configurator.get_model()
params = config.model.params.denoising_network.params

batch= next(iter(dataloader.train_dataloader()))

print("init trainer")
import pytorch_lightning as pl
configurator.create_model_dir()
wandb_logger = configurator.get_logger()
callbacks = configurator.get_callbacks(model=model, dataloader=dataloader)

trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        precision=config.trainer.precision,
        accelerator=config.trainer.accelerator,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
    )

print("Fitting model")
trainer.fit(model, dataloader)

print("Test done")