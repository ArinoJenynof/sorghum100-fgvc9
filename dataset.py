from pathlib import Path
from typing import Any, Optional, Callable, Tuple
from sklearn.preprocessing import LabelEncoder
from torchvision.datasets import VisionDataset
from PIL import Image
import pandas

class Sorghum100(VisionDataset):
	def __init__(
		self,
		root: str,
		train: bool = True,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None
	) -> None:
		super().__init__(root, transform=transform, target_transform=target_transform)
		self.train = train

		self.data_dir = Path(root).resolve()
		self.train_dir = self.data_dir / "train_images"
		self.test_dir = self.data_dir / "test"

		if not (self.data_dir / "train_cultivar_mapping.csv").exists():
			raise RuntimeError(f"Dataset seems missing. Please download it and put it in {root}")

		# There's an extra .DS_Store, drop it
		self.df_train = pandas.read_csv(self.data_dir / "train_cultivar_mapping.csv")
		self.df_train.drop([3329], inplace=True)

		# Can be a simple list, but for consistency let it be DataFrame
		self.df_test = pandas.DataFrame([x.relative_to(self.test_dir) for x in self.test_dir.iterdir() if x.is_file()], columns=["filename"])

		# Leverage sklearn's LabelEncoder for similarly class_to_idx cos i'm lazy :P
		cultivar = self.df_train["cultivar"].unique()
		self.le = LabelEncoder()
		self.le.fit(cultivar)
	
	def __len__(self) -> int:
		if self.train:
			return self.df_train.shape[0]
		return self.df_test.shape[0]
	
	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		if self.train:
			img = Image.open(self.train_dir / self.df_train.iloc[index, 0])
			target = self.le.transform([self.df_train.iloc[index, 1]])

			if self.transform is not None:
				img = self.transform(img)
			if self.target_transform is not None:
				target = self.target_transform(target)
			
			return img, target
		
		img = Image.open(self.test_dir / self.df_test.iloc[index, 0])
		target = None
		if self.transform is not None:
			img = self.transform(img)
		return img, target
