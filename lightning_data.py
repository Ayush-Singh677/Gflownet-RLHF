from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class PromptDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        tokenizer,
        train_size=0.95,
        limit_prompts=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="tokenizer")
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None

    def setup(self, stage=None):
        with open(self.hparams.data_path, "r") as f:
            prompts = f.readlines()
        prompts = [line.rstrip("\n") for line in prompts]
        if self.hparams.limit_prompts is not None:
            prompts = prompts[: self.hparams.limit_prompts]
        num_train = int(len(prompts) * self.hparams.train_size)
        self.train_data = PromptDataset(prompts[:num_train], self.tokenizer)
        self.val_data = PromptDataset(prompts[num_train:], self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=None, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=None, num_workers=0)


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.tokenizer(
            self.prompts[index],
            return_tensors="pt",
        )["input_ids"]
        return prompt