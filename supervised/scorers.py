import torch
import torch.nn as nn
import torch.nn.functional as F

class Scorer(nn.Module):

	def __init__(self, input_size, output_size):

		super(Scorer, self).__init__()

		self.linear_scorer_1 = nn.Linear(input_size, input_size)
		self.linear_scorer_2 = nn.Linear(input_size, output_size)

	def forward(self, attended_instruction, state, other=[]):

		full_input = torch.cat([attended_instruction, state], dim=1) if torch.is_tensor(state) else attended_instruction
		for other_vec in other:
			full_input = torch.cat([full_input, other_vec], dim=1)

		return self.linear_scorer_2(F.relu(self.linear_scorer_1(full_input)))
