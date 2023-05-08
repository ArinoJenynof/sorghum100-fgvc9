# Stolen from ArcFace official implementation (InsightFace) :vvvv
import math
import torch


class ArcFace(torch.nn.Module):
	def __init__(self, s: float = 64.0, margin: float = 0.5):
		super().__init__()
		self.scale = s
		self.margin = margin
		self.cos_m = math.cos(margin)
		self.sin_m = math.sin(margin)
		self.theta = math.cos(math.pi - margin)
		self.sinmm = math.sin(math.pi - margin) * margin
		self.easy_margin = False

	def forward(self, logits: torch.Tensor, labels: torch.Tensor):
		index = torch.where(labels != -1)[0]
		target_logit = logits[index, labels[index].view(-1)]

		with torch.no_grad():
			target_logit.arccos_()
			logits.arccos_()
			final_target_logit = target_logit + self.margin
			logits[index, labels[index].view(-1)] = final_target_logit
			logits.cos_()
		logits = logits * self.s
		return logits


class ArcFaceFC(torch.nn.Module):
	def __init__(self, arcface: ArcFace, embedding_size: int, num_classes: int) -> None:
		super().__init__()
		self.arcface = arcface
		self.embedding_size = embedding_size
		self.cross_entropy = torch.nn.CrossEntropyLoss()
		self.weight = torch.nn.Parameter(torch.normal(0.0, 0.01, [num_classes, embedding_size]))

	def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		labels.squeeze_()
		labels = labels.to(torch.long)

		labels = labels.view(-1, 1)
		weight = self.weight

		norm_embeddings = torch.nn.functional.normalize(images)
		norm_weight_activated = torch.nn.functional.normalize(weight)
		logits = torch.nn.functional.linear(norm_embeddings, norm_weight_activated)

		logits = logits.clamp(-1, 1)
		logits = self.arcface(logits, labels)
		return self.cross_entropy(logits, labels)
