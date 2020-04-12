import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sentencepiece as spm
from utils import *
from attenders import BasicAttender, TimeAttender
from encoders import StateEncoder, InstructionEncoder
from transformers import BertModel, BertTokenizer, T5Tokenizer, T5ForConditionalGeneration
from scorers import Scorer
import sys

class BasicModel(nn.Module):

	def __init__(self, args):

		super(BasicModel, self).__init__()

		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embeddings = nn.Embedding(args["vocab_size"], args["embedding_size"])
		self.instruction_encoder = InstructionEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.state_encoder = StateEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.attender = BasicAttender(args["hidden_size"])

		self.t_scorer = Scorer(args["hidden_size"] * 4, args["template_size"])
		self.o1_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"], args["output_vocab_size"])
		self.o2_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"])


	def forward(self, state, instruction):

		encoded_state = self.state_encoder(self.embeddings, state)
		full_instruction_encoder_output = self.instruction_encoder(self.embeddings, instruction, encoded_state)
		attended_instruction = self.attender(full_instruction_encoder_output, encoded_state)

		q_t = self.t_scorer(attended_instruction, encoded_state)
		q_o1 = self.o1_scorer(attended_instruction, encoded_state, [q_t.detach()])
		q_o2 = self.o2_scorer(attended_instruction, encoded_state, [q_t.detach(), q_o1.detach()])

		return F.log_softmax(q_t, dim=1), F.log_softmax(q_o1, dim=1), F.log_softmax(q_o2, dim=1)

	def eval(self, state, instruction):
		with torch.no_grad():
			t_prob, o1_prob, o2_prob = self.forward(state, instruction)
			t, o1, o2 = torch.argmax(t_prob, dim=1), torch.argmax(o1_prob, dim=1), torch.argmax(o2_prob, dim=1)

			if len(state) == 1:
				return t.item(), o1.item(), o2.item(), t_prob, o1_prob, o2_prob
			return t, o1, o2, t_prob, o1_prob, o2_prob

	def get_embedding(self, idx):
		return self.embeddings(torch.tensor(idx, dtype=torch.long))

	def get_name(self):
		return "basic"

class TimeModel(nn.Module):

	def __init__(self, args):

		super(TimeModel, self).__init__()

		self.args = args
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.embeddings = nn.Embedding(args["vocab_size"], args["embedding_size"])
		self.instruction_encoder = InstructionEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"], basic=False)
		self.state_encoder = StateEncoder(args["embedding_size"], args["hidden_size"], args["spm_path"])
		self.attender = TimeAttender(args["hidden_size"], args["max_number_of_sentences"])

		self.t_scorer = Scorer(args["hidden_size"] * 4, args["template_size"])
		self.o1_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"], args["output_vocab_size"])
		self.o2_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"])

	def forward(self, state, instruction, previous_sentence_attention, previous_word_attention):

		encoded_state = self.state_encoder(self.embeddings, state)

		full_instruction_encoder_output, word_weights, sentence_lengths = self.instruction_encoder(self.embeddings, instruction, encoded_state, previous_word_attention)
		attended_instruction, sentence_weights = self.attender(full_instruction_encoder_output, encoded_state, previous_sentence_attention, sentence_lengths)

		q_t = self.t_scorer(attended_instruction, encoded_state)
		q_o1 = self.o1_scorer(attended_instruction, encoded_state, [q_t.detach()])
		q_o2 = self.o2_scorer(attended_instruction, encoded_state, [q_t.detach(), q_o1.detach()])

		return F.log_softmax(q_t, dim=1), F.log_softmax(q_o1, dim=1), F.log_softmax(q_o2, dim=1), sentence_weights, word_weights

	def eval(self, state, instruction, previous_attention):
		with torch.no_grad():
			t_prob, o1_prob, o2_prob, sentence_weights = self.forward(state, instruction, previous_attention)
			t, o1, o2 = torch.argmax(t_prob, dim=1).item(), torch.argmax(o1_prob, dim=1).item(), torch.argmax(o2_prob, dim=1).item()
			return t, o1, o2, t_prob, o1_prob, o2_prob, sentence_weights

	def get_embedding(self, idx):
		return self.embeddings(torch.tensor(idx, dtype=torch.long))

	def get_name(self):
		return "time"

class TransformerModel(nn.Module):

	def __init__(self, args):

		super(TransformerModel, self).__init__()

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.bert = BertModel.from_pretrained("bert-base-uncased")
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

		self.bert_output_size = self.bert.embeddings.word_embeddings.embedding_dim
		self.t_scorer = Scorer(self.bert_output_size, args["template_size"])
		self.o1_scorer = Scorer(self.bert_output_size + args["template_size"], args["output_vocab_size"])
		self.o2_scorer = Scorer(self.bert_output_size + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"])

	def forward(self, states, instructions, training=True):

		if training:
			self.bert.train()
		else:
			self.bert.eval()

		def get_inputs():
			input_tokens = []
			segment_ids = []
			attention_masks = []
			for pair in zip(states, instructions):
				encoded_dict = self.tokenizer.encode_plus(pair[0], pair[1], max_length=500, pad_to_max_length=True)
				input_tokens.append(encoded_dict["input_ids"])
				segment_ids.append(encoded_dict['token_type_ids'])
				attention_masks.append(encoded_dict['attention_mask'])
			return torch.tensor(input_tokens).to(self.device), torch.tensor(segment_ids).to(self.device), torch.tensor(attention_masks).to(self.device)

		input_tensor, segment_tensor, attention_mask_tensor = get_inputs()
		all_hidden_states, all_attentions = self.bert(input_tensor, token_type_ids=segment_tensor, attention_mask=attention_mask_tensor)

		q_t = self.t_scorer(all_hidden_states[:, -1, :], None)
		q_o1 = self.o1_scorer(all_hidden_states[:, -1, :], None, [q_t.detach()])
		q_o2 = self.o2_scorer(all_hidden_states[:, -1, :], None, [q_t.detach(), q_o1.detach()])

		return F.log_softmax(q_t, dim=1), F.log_softmax(q_o1, dim=1), F.log_softmax(q_o2, dim=1)

	def eval(self, state, instruction):

		with torch.no_grad():
			t, o1, o2 = torch.argmax(t_prob, dim=1), torch.argmax(o1_prob, dim=1), torch.argmax(o2_prob, dim=1)

			t = torch.argmax(t_prob, dim=1)
			if len(state) == 1:
				return t.item(), o1.item(), o2.item(), t_prob, o1_prob, o2_prob

			return t, o1, o2, t_prob, o1_prob, o2_prob

	def get_name(self):
		return "transformer"

class TranslationTransformerModel(nn.Module):

	def __init__(self, args):

		super(TranslationTransformerModel, self).__init__()

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
		self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
		self.args = args

		#self.o1_scorer = Scorer(self.t5.config.hidden_size + args["template_size"], args["output_vocab_size"], d=64)
		#self.o2_scorer = Scorer(self.t5.config.hidden_size + args["template_size"], args["output_vocab_size"], d=64)

		self.o_criterion = nn.NLLLoss()

	def forward(self, instructions, states, actions=None, o1s=None, o2s=None, training=True):

		if training:
			self.t5.train()
		else:
			self.t5.eval()

		def get_inputs(instruction, states, actions):
			instruction_dict = self.tokenizer.encode_plus("instruction_interpet: " + instruction + "</s>", max_length=400, pad_to_max_length=True)
			instruction_input_tokens = instruction_dict["input_ids"]
			instruction_attention_masks = instruction_dict["attention_mask"]

			state_input_tokens = []
			state_attention_masks = []			
			for state in states:
				state_dict = self.tokenizer.encode_plus("state_interpret: " + state, max_length=200, pad_to_max_length=True)
				state_input_tokens.append(state_dict["input_ids"])
				state_attention_masks.append(state_dict['attention_mask'])

			return torch.tensor([instruction_input_tokens]), torch.tensor([instruction_attention_masks]), torch.tensor(state_input_tokens), \
				torch.tensor(state_attention_masks), torch.tensor([actions])

		def get_idxs(vals):
			idxs = []
			val_select = []
			for i, o in enumerate(vals):
				if o != None:
					idxs.append(i)
					val_select.append(o)
			return torch.tensor(idxs, dtype=torch.long), torch.tensor(val_select, dtype=torch.long)

		instruction_input_tokens, instruction_attention_masks, state_input_tokens, state_attention_masks, action_tensor = get_inputs(instructions, states, actions)
		instruction_loss, instruction_logits, instruction_hidden = self.t5(input_ids=instruction_input_tokens, attention_mask=instruction_attention_masks, decoder_input_ids=action_tensor, lm_labels=action_tensor)
		#_, state_logits, _ = self.t5(input_ids=state_input_tokens, attention_mask=state_attention_masks, lm_labels=action_tensor.permute(1, 0))

		q_t = instruction_logits[:, :, 0:self.args["template_size"]]
		q_t_probs = F.log_softmax(q_t, dim=2)

		q_o1 = instruction_logits[:, :, self.args["template_size"]:(self.args["template_size"] + self.args["output_vocab_size"])]
		o1_idxs, o1s_select = get_idxs(o1s)
		q_o1_select = torch.index_select(q_o1.squeeze(0), 0, o1_idxs)
		q_o1_probs = F.log_softmax(q_o1_select, dim=1)
		o1_loss = self.o_criterion(q_o1_probs, o1s_select) if (o1s != None and q_o1_select.size()[0] != 0) else None

		q_o2 = instruction_logits[:, :, (self.args["template_size"] + self.args["output_vocab_size"]):(self.args["template_size"] + self.args["output_vocab_size"] + self.args["output_vocab_size"])]
		o2_idxs, o2s_select = get_idxs(o2s)
		q_o2_select = torch.index_select(q_o2.squeeze(0), 0, o2_idxs)
		q_o2_probs = F.log_softmax(q_o2_select, dim=1)
		o2_loss = self.o_criterion(q_o2_probs, o2s_select) if (o2s != None and q_o2_select.size()[0] != 0) else None

		return (q_t_probs, instruction_loss), (q_o1_probs, o1_loss), (q_o2_probs, o2_loss)

	def eval(self, instructions, states, actions=None, o1s=None, o2s=None,):
		with torch.no_grad():
			return self.forward(instructions, states, actions, o1s, o2s, training=False)


	def get_name(self):
		return "translate"











