from tqdm import tqdm, tqdm_notebook


class ProgressBar:
    """
    Pretty progress bar for training
    """

    def __init__(self, epoch, total_epochs, data_loader, is_notebook=False):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.loss = 0
        _tqdm = tqdm_notebook if is_notebook else tqdm
        self.progress = tqdm(
            iterable=iter(data_loader),
            leave=False,
            desc=f"epoch {epoch + 1} / {total_epochs}",
            total=len(data_loader),
            miniters=0,
        )

    def update(self, loss):
        self.loss = loss
        self.progress.set_postfix(loss=loss, refresh=False)

    def __iter__(self):
        return self.progress.__iter__()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.progress.close()
        print(f"epoch {self.epoch + 1} / {self.total_epochs}, loss: {self.loss:0.4f}")
