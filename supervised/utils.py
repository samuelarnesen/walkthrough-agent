import torch
import numpy as np
from jericho import *
from jericho.template_action_generator import TemplateActionGenerator


def convert_batch_to_tokens(batch, max_sequence, device, sp):
	np_list = []

	for example in batch:
		np_list.append(convert_to_tokens(example, max_sequence, sp))

	return torch.tensor(data=np.asarray(np_list, dtype=np.float16), dtype=torch.long, device=device)

def convert_to_tokens(text, max_sequence, sp):
	split_text = text.split("|")
	token_array = np.zeros([len(split_text), max_sequence])
	for i, sequence in enumerate(split_text):
		enc_seq = sp.encode_as_ids("<s>" + sequence + "</s>")
		for j, token in enumerate(enc_seq):
			token_array[i][j] = np.int32(token)
	return token_array

def tokenize_sentence(sp, sentence):
	return sp.encode_as_ids(sentence)

def get_vocab_size(sp):
	return len(sp)

def load_indices(file_path):
	train_idxs = []
	val_idxs = []
	test_idxs =  []
	with open(file_path) as f:
		current = ""
		for line in f.readlines():
			if "train" in line or "val" in line or "test" in line:
				current = line.strip(" \n:")
				continue
			if current == "train":
				train_idxs.append(int(line.strip(" \n")))
			elif current == "val":
				val_idxs.append(int(line.strip(" \n")))
			elif current == "test":
				test_idxs.append(int(line.strip(" \n")))
	return train_idxs, val_idxs, test_idxs

def dump_indices(file_path, train_idxs, val_idxs, test_idxs):

	with open(file_path, "w") as f:
		print("train", file=f)
		for idx in train_idxs:
			print(idx, file=f)

		print("val", file=f)
		for idx in val_idxs:
			print(idx, file=f)

		print("test", file=f)
		for idx in test_idxs:
			print(idx, file=f)

def create_templates_list(rom_paths, additional_templates_path=None):

	def count_obj(word):
		return word.count("OBJ") + (len(word.split()) / 10)

	templates = []
	for rom_path in rom_paths:
		binding = jericho.load_bindings(rom_path)
		game_templates = TemplateActionGenerator(binding).templates
		for template in game_templates:
			if template not in templates:
				if template != "push OBJ OBJ": # this was messing stuff up
					templates.append(template)

	if additional_templates_path != None:
		with open(additional_templates_path) as f:
			for line in f.readlines():
				if line not in templates:
					templates.append(line.rstrip("\n"))

	return list(reversed(sorted(templates, key=count_obj)))

def load_vocab(rom_paths, additional_words_path=None):

	vocab = {}
	reverse = {}
	for rom_path in rom_paths:
		env = FrotzEnv(rom_path)
		for word in env.get_dictionary():
			if str(word) not in reverse:
				reverse[str(word)] = len(vocab) + 2
				vocab[len(vocab) + 2] = str(word)

	if additional_words_path != None:
		with open(additional_words_path) as f:
			for line in f.readlines():
				word = line.strip("\n")
				reverse[word] = len(vocab) + 2
				vocab[len(vocab) + 2] = word
	return vocab, reverse

def clip_words(sentence, max_length):
	split_sentence = sentence.split(" ")
	for i, word in enumerate(split_sentence):
		split_sentence[i] = word[0:min(len(word), max_length)]
	return " ".join(split_sentence)

def are_equivalent(reconstruction, original):
	clipped_original = clip_words(original, 6)
	if reconstruction.lower() == clipped_original.lower():
		return True
	return "".join(clipped_original.split("\"")).lower() == reconstruction.lower()

def are_cmd_equivalent(reconstruction, original):

	if are_equivalent(reconstruction, original):
		return True

	if "CMD" not in reconstruction and "," not in original:
		return False

	original_subbed = clip_words(original, 6)
	if "CMD" in reconstruction:
		original_subbed = original_subbed[0:reconstruction.index("CMD")] + "CMD"
	if "," in original_subbed:
		original_subbed = original_subbed[original_subbed.index(",") + 1:].strip(" ")
	return are_equivalent(reconstruction, original_subbed)

def extract_object(word):

	split_word = word.split()
	if len(split_word) <= 1:
		return word

	if split_word[1] in ["of", "the"]:
		return split_word[0]

	else:
		return split_word[1]