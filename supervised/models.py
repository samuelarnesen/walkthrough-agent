import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sentencepiece as spm
from utils import *
from attenders import BasicAttender, TimeAttender
from encoders import StateEncoder, InstructionEncoder
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
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

		self.t_scorer = Scorer(args["hidden_size"] * 4, args["template_size"], d=32)
		self.o1_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"], args["output_vocab_size"], d=64)
		self.o2_scorer = Scorer((args["hidden_size"] * 4) + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"], d=64)

	def forward(self, state, instruction, previous_sentence_attention, previous_word_attention):

		encoded_state = self.state_encoder(self.embeddings, state)

		full_instruction_encoder_output, word_weights, sentence_lengths = self.instruction_encoder(self.embeddings, instruction, encoded_state, previous_word_attention)
		attended_instruction, sentence_weights = self.attender(full_instruction_encoder_output, encoded_state, previous_sentence_attention, sentence_lengths)

		q_t = self.t_scorer(attended_instruction, encoded_state)
		q_o1 = self.o1_scorer(attended_instruction, encoded_state, [q_t.clone().detach()])
		q_o2 = self.o2_scorer(attended_instruction, encoded_state, [q_t.clone().detach(), q_o1.clone().detach()])

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
		self.t_scorer = Scorer(self.bert_output_size, args["template_size"], d=32)
		self.o1_scorer = Scorer(self.bert_output_size + args["template_size"], args["output_vocab_size"], d=64)
		self.o2_scorer = Scorer(self.bert_output_size + args["template_size"] + args["output_vocab_size"], args["output_vocab_size"], d=64)

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

	def eval(self, states, instructions):

		with torch.no_grad():
			t_prob, o1_prob, o2_prob = self(states, instructions, training=False)
			t, o1, o2 = torch.argmax(t_prob, dim=1), torch.argmax(o1_prob, dim=1), torch.argmax(o2_prob, dim=1)
			if len(states) == 1:
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

		special_tokens_dict = {
			"eos_token": "</s>",
			"bos_token": "<s>",
			"unk_token": "<unk>",
			"pad_token": "<pad>"
		}
		self.tokenizer.add_special_tokens(special_tokens_dict)

	def forward(self, instruction, actions, repeater=None, command_mode=True, training=False):
		if training:
			self.t5.train()
		else:
			self.t5.eval()

		if command_mode:
			return self.predict_commands(instruction, actions, training)
		else:
			return self.predict_criteria_met(repeater)

	# i should change this anyways
	def eval(self, instruction, actions):
		with torch.no_grad():
			return self.forward(instruction, actions, False)


	def predict_commands(self, instruction, actions, training=True):

		instruction_dict = self.tokenizer.encode_plus("<s> instruction_interpet: " + instruction + "</s>", max_length=400, pad_to_max_length=True, return_tensors="pt")
		instruction_ids = instruction_dict["input_ids"]
		instruction_masks = instruction_dict["attention_mask"]
		action_ids = self.tokenizer.encode("<pad> " + actions + " </s>", return_tensors="pt")
		decoder_input_ids = action_ids[:, :-1]
		lm_labels = action_ids[:, 1:].clone()

		loss, logits, hidden = self.t5(input_ids=instruction_ids, attention_mask=instruction_masks, decoder_input_ids=decoder_input_ids, lm_labels=lm_labels)
		
		pred_probs = F.softmax(logits, dim=2)
		preds = torch.argmax(pred_probs.squeeze(0), dim=1).squeeze(0)
		reconstructed_string = self.tokenizer.decode(preds)

		return pred_probs, loss, reconstructed_string

	def predict_criteria_met(self, repeater):

		state_ids = []
		state_masks = []
		token_type_ids = []

		text_pairs = [("<s> classify_state: state: " + state + "</s>", "condition: " + repeater.terminal_condition + "</s>") for state in repeater.states]
		state_dict = self.tokenizer.batch_encode_plus(text_pairs, max_length=400, pad_to_max_length=True, return_tensors="pt")

		actions = ["<pad> not_satisfied </s>" for i in range(len(repeater.states))]
		actions[-1] = "<pad> is_satisfied </s>"

		action_ids = self.tokenizer.batch_encode_plus(actions, return_tensors="pt")["input_ids"]

		decoder_input_ids = action_ids[:, :-1]
		lm_labels = action_ids[:, 1:].clone()

		loss, logits, hidden = self.t5(input_ids=state_dict["input_ids"], attention_mask=state_dict["attention_mask"], \
			decoder_input_ids=decoder_input_ids, lm_labels=lm_labels)

		pred_probs = F.softmax(logits, dim=2)
		preds = torch.argmax(pred_probs, dim=2)
		reconstructed_strings = []
		for i in range(len(repeater.states)):
			reconstructed_strings.append(self.tokenizer.decode(preds[i, :]))

		return pred_probs, loss, " | ".join(reconstructed_strings)



	def get_name(self):
		return "translate"













