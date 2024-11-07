import pytorch_lightning as pl


class DummyLightningModule(pl.LightningModule):
    """
    def fit(self):
        configure_callbacks()
        if local_rank == 0:
            prepare_data()
        setup("fit")
        configure_model()
        configure_optimizers()
        on_fit_start()

        # the sanity check runs here
        on_train_start()
        for epoch in epochs:
            fit_loop()
        on_train_end()
        on_fit_end()
        teardown("fit")

    def fit_loop():
        torch.set_grad_enabled(True)
        on_train_epoch_start()
        for batch in train_dataloader():
            on_train_batch_start()
            on_before_batch_transfer()
            transfer_batch_to_device()
            on_after_batch_transfer()
            out = training_step()
            on_before_zero_grad()
            optimizer_zero_grad()
            on_before_backward()
            backward()
            on_after_backward()
            on_before_optimizer_step()
            configure_gradient_clipping()
            optimizer_step()
            on_train_batch_end(out, batch, batch_idx)
            if should_check_val:
                val_loop()
        on_train_epoch_end()

    def val_loop():
        on_validation_model_eval()  # calls `model.eval()`
        torch.set_grad_enabled(False)
        on_validation_start()
        on_validation_epoch_start()
        for batch_idx, batch in enumerate(val_dataloader()):
            on_validation_batch_start(batch, batch_idx)
            batch = on_before_batch_transfer(batch)
            batch = transfer_batch_to_device(batch)
            batch = on_after_batch_transfer(batch)
            out = validation_step(batch, batch_idx)
            on_validation_batch_end(out, batch, batch_idx)
        on_validation_epoch_end()
        on_validation_end()

        # set up for train
        on_validation_model_train()  # calls `model.train()`
        torch.set_grad_enabled(True)
    """

    def __init__(self):
        "dummy_init = 0"

    def backward(self, loss):
        """
        Called to perform backward on the loss returned in training_step()
        """
        loss.backward()

    def on_before_backward(self):
        """
        Called before loss.backward()
        """
        return None

    def on_after_backward(self):
        """
        Called after loss.backward() and before optimizers are stepped
        """
        return None

    def on_before_zero_grad(self, optimizer):
        """
        Called after training_step() and before optimizer.zero_grad().
        Called in the training loop after taking an optimizer step and before zeroing grads.
        Good place to inspect weight information with weights updated.
        """
        return None

    def on_fit_start(self):
        """
        Called at the very beginning of fit.
        If on DDP it is called on every process.
        """
        return None

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process.
        """
        return None

    def on_load_checkpoint(self, checkpoint):
        """
        Called by Lightning to restore your model.
        If you saved something with on_save_checkpoint() this is your chance to restore this.
        """
        self.something_cool_i_want_to_save = checkpoint["something_cool_i_want_to_save"]

    def on_save_checkpoint(self, checkpoint):
        """
        Called by Lightning when saving a checkpoint to give you a chance to store anything else you might want to save.
        """
        checkpoint["something_cool_i_want_to_save"] = "my_cool_pickable_object"

    def load_from_checkpoint(
        self,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=None,
        **kwargs,
    ):
        """
        Primary way of loading a model from a checkpoint.
        When Lightning saves a checkpoint it stores the arguments passed to __init__ in the checkpoint under "hyper_parameters".
        Returns LightningModule instance with loaded weights and hyperparameters (if available).
        """

    def on_train_start(self):
        """
        Called at the beginning of training after sanity check.
        """
        return None

    def on_train_end(self):
        """
        Called at the end of training before logger experiment is closed.
        """
        return None

    def on_validation_start(self):
        """
        Called at the beginning of validation.
        """
        return None

    def on_validation_end(self):
        """
        Called at the end of validation.
        """
        return None

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """
        Called in the test loop before anything happens for that batch.
        """
        return None

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called in the test loop after the batch.
        """
        return None

    def on_test_epoch_start(self):
        """
        Called in the test loop at the very beginning of the epoch.
        """
        return None

    def on_test_epoch_end(self):
        """
        Called in the test loop at the very end of the epoch.
        """
        return None

    def on_test_start(self):
        """
        Called at the beginning of testing.
        """
        return None

    def on_test_end(self):
        """
        Called at the end of testing.
        """
        return None

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """
        Called in the predict loop before anything happens for that batch.
        """
        return None

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called in the predict loop after the batch.
        """
        return None

    def on_predict_epoch_start(self):
        """
        Called at the beginning of predicting.
        """
        return None

    def on_predict_epoch_end(self):
        """
        Called at the end of predicting.
        """
        return None

    def on_predict_start(self):
        """
        Called at the beginning of predicting.
        """
        return None

    def on_predict_end(self):
        """
        Called at the end of predicting.
        """
        return None

    def on_train_batch_start(self, batch, batch_idx):
        """
        Called in the training loop before anything happens for that batch.
        If you return -1 here, you will skip training for the rest of the current epoch.
        """
        return int

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Called in the training loop after the batch.
        """
        return None

    def on_train_epoch_start(self):
        """
        Called in the training loop at the very beginning of the epoch.
        """
        return None

    def on_train_epoch_end(self):
        """
        Called in the training loop at the very end of the epoch.
        """
        return None

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        """
        Called in the validation loop before anything happens for that batch.
        """
        return None

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called in the validation loop after the batch.
        """
        return None

    def on_validation_epoch_start(self):
        """
        Called in the validation loop at the very beginning of the epoch.
        """
        return None

    def on_validation_epoch_end(self):
        """
        Called in the validation loop at the very end of the epoch.
        """
        return None

    def configure_model(self):
        """
        Hook to create modules in a strategy and precision aware context.
        This is particularly useful for when using sharded strategies (FSDP and DeepSpeed),
        where we’d like to shard the model instantly to save memory and initialization time.
        For non-sharded strategies, you can choose to override this hook or to initialize your model under the init_module() context manager.
        This hook is called during each of fit/val/test/predict stages in the same process,
        so ensure that implementation of this hook is idempotent, i.e., after the first time the hook is called, subsequent calls to it should be a no-op.
        """
        return None

    def on_validation_model_eval(self):
        """
        Called when the validation loop starts.
        The validation loop by default calls .eval() on the LightningModule before it starts.
        """
        return None

    def on_validation_model_train(self):
        """
        Called when the validation loop ends.
        The validation loop by default restores the training mode of the LightningModule to what it was before starting validation.
        """
        return None

    def on_test_model_eval(self):
        """
        Called when the test loop starts.
        The test loop by default calls .eval() on the LightningModule before it starts.
        """
        return None

    def on_test_model_train(self):
        """
        Called when the test loop ends.
        The test loop by default restores the training mode of the LightningModule to what it was before starting testing.
        """
        return None

    def on_before_optimizer_step(self, optimizer):
        """
        Called before optimizer.step().
        If using gradient accumulation, the hook is called once the gradients have been accumulated.
        If using AMP, the loss will be unscaled before calling this hook.
        If clipping gradients, the gradients will not have been clipped yet.
        """
        return None

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """
        Perform gradient clipping for the optimizer parameters. Called before optimizer_step().
        """
        return None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """
        Override this method to adjust the default way the Trainer calls the optimizer.
        By default, Lightning calls step() and zero_grad().
        This method (and zero_grad()) won’t be called during the accumulation phase when Trainer(accumulate_grad_batches != 1).
        Overriding this hook has no benefit with manual optimization.
        """
        return None

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """
        Override this method to change the default behaviour of optimizer.zero_grad().
        """
        return None

    def prepare_data(self):
        """
        Use this to download and prepare data. Downloading and saving data with multiple processes (distributed settings) will result in corrupted data.
        Lightning ensures this method is called only within a single process, so you can safely add your downloading logic within.
        """
        return None

    def setup(self, stage):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something about them.
        This hook is called on every process when using DDP.
        """
        return None

    def teardown(self, stage):
        """
        Called at the end of fit (train + validate), validate, test, or predict.
        """
        return None

    def train_dataloader(self):
        """
        An iterable or collection of iterables specifying training samples.
        The dataloader you return will not be reloaded unless you set reload_dataloaders_every_n_epochs to a positive integer.
        For data processing use the following pattern: :rtype: Any
                download in prepare_data()
                process and split in setup()
        However, the above are only necessary for distributed processing.
        /!\ do not assign state in prepare_data.
                fit()
                prepare_data()
                setup()
        """
        return None

    def val_dataloader(self):
        """
        An iterable or collection of iterables specifying validation samples.
        The dataloader you return will not be reloaded unless you set reload_dataloaders_every_n_epochs to a positive integer.
        It’s recommended that all data downloads and preparation happen in prepare_data(). :rtype: Any
            fit()
            validate()
            prepare_data()
            setup()
        """
        return None

    def test_dataloader(self):
        """
        An iterable or collection of iterables specifying test samples.
        For more information about multiple dataloaders, see this section.
        For data processing use the following pattern: :rtype: Any
                download in prepare_data()
                process and split in setup()
        However, the above are only necessary for distributed processing.
        /!\ do not assign state in prepare_data.
                fit()
                prepare_data()
                setup()
        """
        return None

    def predict_dataloader(self):
        """
        An iterable or collection of iterables specifying prediction samples.
        For more information about multiple dataloaders, see this section.
        It’s recommended that all data downloads and preparation happen in prepare_data().
            predict()
            prepare_data()
            setup()
        Returns a torch.utils.data.DataLoader or a sequence of them specifying prediction samples.
        """
        return None

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Override this hook if your DataLoader returns tensors wrapped in a custom data structure.
        The data types listed below (and any arbitrary nesting of them) are supported out of the box:
            torch.Tensor or anything that implements .to(…)
            list
            dict
            tuple
        For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, …).
        This hook should only transfer the data and not modify it, nor should it move the data to any other
        device than the one passed in as argument (unless you know what you are doing).
        Returns a reference to the data on the new device.
        """
        return None

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """
        Override to alter or apply batch augmentations to your batch before it is transferred to the device.
        Returns a batch of data
        """
        return None

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Override to alter or apply batch augmentations to your batch before it is transferred to the device.
        Returns a batch of data
        """
        return None
