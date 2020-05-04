import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Scorer(nn.Module):

	def __init__(self, input_size, output_size, d=None):

		super(Scorer, self).__init__()

		if d == None:
			d = input_size

		self.linear_scorer_1 = nn.Linear(input_size, d)
		self.linear_scorer_2 = nn.Linear(d, output_size)

	def forward(self, attended_instruction, state, other=[], dim=1):

		full_input = torch.cat([attended_instruction, state], dim=dim) if torch.is_tensor(state) else attended_instruction
		for other_vec in other:
			full_input = torch.cat([full_input, other_vec], dim=dim)

		return self.linear_scorer_2(F.relu(self.linear_scorer_1(full_input)))
