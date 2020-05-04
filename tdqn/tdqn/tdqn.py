import time
import math, random
import numpy as np
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import logger
import copy, sys, pickle, random
sys.path.append("../../supervised/")
from parse_walkthrough import Walkthrough, SuperWalkthrough
from utils import *

from replay import *
from schedule import *
from models import TDQN

from env import *
import jericho
from jericho.template_action_generator import TemplateActionGenerator

import sentencepiece as spm


def configure_logger(log_dir):
    logger.configure(log_dir, format_strs=['log'])
    global tb
    tb = logger.Logger(log_dir, [logger.make_output_format('csv', log_dir),
                                 logger.make_output_format('stdout', log_dir)])
    global log
    log = logger.log



class TDQN_Trainer(object):
    def __init__(self, args):
        configure_logger(args.output_dir)
        log(args)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.args = args

        self.log_freq = args.log_freq
        self.update_freq = args.update_freq_td
        self.update_freq_tar = args.update_freq_tar
        self.filename = 'tdqn'

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.binding = jericho.load_bindings(args.rom_path)
        self.vocab_act, self.vocab_act_rev = self.load_vocab_act(args.rom_path)
        vocab_size = len(self.sp)
        vocab_size_act = len(self.vocab_act.keys())

        self.templates = create_templates_list([args.rom_path], args.temp_path)
        self.template_size = len(self.templates)

        if args.replay_buffer_type == 'priority':
            self.replay_buffer = PriorityReplayBuffer(int(args.replay_buffer_size))
            self.wt_buffer = PriorityReplayBuffer(1000)
        elif args.replay_buffer_type == 'standard':
            self.replay_buffer = ReplayBuffer(int(args.replay_buffer_size))
            self.wt_buffer = ReplayBuffer(1000)

        self.model = TDQN(args, self.template_size, vocab_size, vocab_size_act).to(self.device)
        self.target_model = TDQN(args, self.template_size, vocab_size, vocab_size_act).to(self.device)

        if args.pretrained_path != None:
            pretrained = torch.load(args.pretrained_path)

            self.model = self.load_model(self.model, pretrained["model"])
            self.target_model = self.load_model(self.target_model, pretrained["target"])

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.num_steps = args.steps
        self.batch_size = args.batch_size
        self.gamma = args.gamma

        self.rho = args.rho

        self.bce_loss = nn.BCELoss()

        self.use_walkthrough = args.use_walkthrough
        self.walkthrough = None
        if self.use_walkthrough:
            #swt = SuperWalkthrough(args.walkthrough_filename, args.rom_path)
            with open(args.walkthrough_filename, "rb") as f:
                swt = pickle.load(f)
                self.walkthrough_actions = swt.get_actions()
                self.walkthrough_states = swt.get_state_descriptions()
                self.possible_actions = swt.get_valid_actions()

    def load_model(self, model, pretrained):
        model.embeddings = pretrained.embeddings
        model.state_network = pretrained.state_network
        model.t_scorer = pretrained.t_scorer
        model.o1_scorer = pretrained.o1_scorer
        model.o2_scorer = pretrained.o2_scorer
        return model


    def load_vocab_act(self, rom_path):
        #loading vocab directly from Jericho
        env = FrotzEnv(rom_path)
        vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
        vocab[0] = ' '
        vocab[1] = '<s>'
        vocab_rev = {v: idx for idx, v in vocab.items()}
        env.close()
        return vocab, vocab_rev

    def state_rep_generator(self, state_description):

        remove = ['=', '-', '\'', ':', '[', ']', 'eos', 'EOS', 'SOS', 'UNK', 'unk', 'sos', '<', '>']
        for rm in remove:
            state_description = state_description.replace(rm, '')

        state_description = state_description.split('|')

        ret = [self.sp.encode_as_ids('<s>' + s_desc + '</s>') for s_desc in state_description]

        return pad_sequences(ret, maxlen=self.args.max_seq_len)

    def plot(self, frame_idx, rewards, losses, completion_steps):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('frame %s. steps: %s' % (frame_idx, np.mean(completion_steps[-10:])))
        plt.plot(completion_steps)
        plt.subplot(133)
        plt.title('loss-lstm-dqn')
        plt.plot(losses)
        # txt = "Gamma:" + str(self.gamma) + ", Num Frames:" + str(self.num_frames) + ", E Decay:" + str(epsilon_decay)
        plt.figtext(0.5, 0.01, self.filename, wrap=True, horizontalalignment='center', fontsize=12)
        # plt.show()
        fig.savefig('plots/' + self.filename + '_' + str(frame_idx) + '.png')

    def compute_td_loss(self):

        state, action, reward, next_state, done, valid = self.replay_buffer.sample(self.batch_size, self.rho)
        action = torch.LongTensor(action).to(self.device)
        state = torch.LongTensor(state).permute(1, 0, 2).to(self.device)
        next_state = torch.LongTensor(next_state).permute(1, 0, 2).detach().to(self.device)
        template_targets = torch.stack([v[0] for v in valid]).to(self.device)
        obj_targets = torch.stack([v[1] for v in valid]).to(self.device)

        decode_steps = []
        for t in action[:, 0]:
            decode_steps.append(self.templates[t.item()].count('OBJ'))

        template = action[:, 0]
        object1 = action[:, 1]
        object2 = action[:, 2]
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(1 * done).to(self.device)

        o1_mask, o2_mask = [0] * self.batch_size, [0] * self.batch_size
        for d, st in enumerate(decode_steps):
            if st > 1:
                o1_mask[d] = 1
                o2_mask[d] = 1
            elif st == 1:
                o1_mask[d] = 1

        o1_mask, o2_mask = torch.FloatTensor(o1_mask).to(self.device), torch.FloatTensor(o2_mask).to(self.device)

        self.model.flatten_parameters()
        q_t, q_o1, q_o2 = self.model(state)

        supervised_loss = self.bce_loss(F.softmax(q_t, dim=1), template_targets)+\
                          self.bce_loss(F.softmax(q_o1, dim=1), obj_targets)+\
                          self.bce_loss(F.softmax(q_o2, dim=1), obj_targets)
        tb.logkv_mean('SupervisedLoss', supervised_loss.item())

        self.target_model.flatten_parameters()
        next_q_t, next_q_o1, next_q_o2 = self.target_model(next_state)

        q_t = q_t.gather(1, template.unsqueeze(1)).squeeze(1)
        q_o1 = q_o1.gather(1, object1.unsqueeze(1)).squeeze(1)
        q_o2 = q_o2.gather(1, object2.unsqueeze(1)).squeeze(1)

        next_q_t = next_q_t.max(1)[0]
        next_q_o1 = next_q_o1.max(1)[0]
        next_q_o2 = next_q_o2.max(1)[0]

        td_loss = F.smooth_l1_loss(q_t, (reward + self.gamma * next_q_t).detach()) +\
                  F.smooth_l1_loss(q_o1 * o1_mask, o1_mask * (reward + self.gamma * next_q_o1).detach()) +\
                  F.smooth_l1_loss(q_o2 * o2_mask, o2_mask * (reward + self.gamma * next_q_o2).detach())

        tb.logkv_mean('TDLoss', td_loss.item())
        loss = td_loss + supervised_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss

    def tmpl_to_str(self, template_idx, o1, o2):
        o1_word = o1 if type(o1) == type("") else self.vocab_act[o1]
        o2_word = o2 if type(o2) == type("") else self.vocab_act[o2]

        template_str = self.templates[template_idx]
        holes = template_str.count('OBJ')
        assert holes <= 2
        if holes <= 0:
            return template_str
        elif holes == 1:
            return template_str.replace('OBJ', o1_word)
        else:
            return template_str.replace('OBJ', o1_word, 1)\
                               .replace('OBJ', o2_word, 1)


    def generate_targets_multilabel(self, valid_acts):
        template_targets = torch.zeros([self.template_size])
        obj_targets = torch.zeros([len(self.vocab_act.keys())])

        for act in valid_acts:

            if type(act) == type(""):
                template_id, o1_id, o2_id = self.identify_components(act)
                template_targets[template_id] = 1
                if o1_id != None:
                    obj_targets[o1_id] = 1
                if o2_id != None:
                    obj_targets[o2_id] = 1

            else:
                template_targets[act.template_id] = 1
                for obj_id in act.obj_ids:
                    obj_targets[obj_id] = 1

        return template_targets, obj_targets


    def train(self):
        start = time.time()

        env = JerichoEnv(self.args.rom_path, self.vocab_act_rev,
                         self.args.env_step_limit, templates=self.templates)
        env.create()

        episode = 1
        state_text, info = env.reset()
        state_rep = self.state_rep_generator(state_text)
    
        walkthrough_idx = 0
        start_score = 0

        for frame_idx in range(1, self.num_steps + 1):
            found_valid_action = False
            while not found_valid_action:
                if self.use_walkthrough and episode == 1:
                    action_str = self.walkthrough_actions[walkthrough_idx]
                    template, o1, o2 = self.identify_components(action_str)
                    action = [template, o1 if o1 != None else 0, o2 if o2 != None else 0]

                    og_nst, reward, done, info = env.walkthrough_step(action_str)
                    next_state_text = self.walkthrough_states[walkthrough_idx + 1]

                    info["valid"] = self.possible_actions[walkthrough_idx]
                    walkthrough_idx += 1
                    found_valid_action = True
                else:

                    templates, o1s, o2s, q_ts, q_o1s, q_o2s = self.model.poly_act(state_rep)
                    for template, o1, o2, q_t, q_o1, q_o2 in zip(templates, o1s, o2s, q_ts, q_o1s, q_o2s):
                        action = [template, o1, o2]
                        action_str = self.tmpl_to_str(template, o1, o2)
                        next_state_text, reward, done, info = env.step(action_str)
                        if info['action_valid'] == True:
                            found_valid_action = True
                            break

            valid_acts = info['valid']
            template_targets, obj_targets = self.generate_targets_multilabel(valid_acts)

            next_state_rep = self.state_rep_generator(next_state_text)
            self.replay_buffer.push(state_rep, action, reward, next_state_rep,
                                    done, (template_targets, obj_targets))
            state_text = next_state_text
            state_rep = next_state_rep

            if done:
                score = info['score']
                if episode % 100 == 0:
                    if self.use_walkthrough:
                        log('Episode {} Score {}\n'.format(episode, score - start_score))
                    else:
                        log('Episode {} Score {}\n'.format(episode, score))
                tb.logkv_mean('EpisodeScore', score - start_score)

                state_text, info = env.reset()

                if self.use_walkthrough:
                    start_pt = random.choice(range(0, int((3/4) * len(self.walkthrough_states))))
                    if start_pt == 0:
                        state_rep = self.state_rep_generator(state_text)
                    else:
                        for i in range(0, start_pt):
                            ob, _, done, info = env.walkthrough_step(self.walkthrough_actions[i])
                        state_rep = self.state_rep_generator(self.walkthrough_states[i + 1])
                        start_score = info["score"]
                    valid_acts = self.possible_actions[start_pt - 1] if start_pt > 0 else [self.walkthrough_actions[0]]
                    env.reset_steps()

                episode += 1

            if len(self.replay_buffer) > self.batch_size:
                if frame_idx % self.update_freq == 0:
                    loss = self.compute_td_loss()
                    tb.logkv_mean('Loss', loss.item())

            if frame_idx % self.update_freq_tar == 0:
                self.target_model = copy.deepcopy(self.model)
                parameters = {
                    'model': self.model,
                    'target': self.target_model,
                    'replay_buffer': self.replay_buffer
                }
                torch.save(parameters, pjoin(self.args.output_dir, self.filename + '_finalv2.pt'))

            if frame_idx % self.log_freq == 0:
                tb.logkv('Step', frame_idx)
                tb.logkv('FPS', int(frame_idx/(time.time()-start)))
                tb.dumpkvs()

        env.close()

        parameters = {
            'model': self.model,
            'target': self.target_model,
            'replay_buffer': self.replay_buffer
        }
        torch.save(parameters, pjoin(self.args.output_dir, self.filename + '_finalv2.pt'))

    def identify_components(self, action):

        lookarounds = ["with", "to", "from", "at", "in", "under", "into"]
        lookaheads = "".join("(?!\\b" + la + "\\b)" for la in lookarounds)
        lookbehinds = "".join("(?<!\\b" + la + "\\b)" for la in lookarounds)

        regex_string = (lookaheads + "(\w+(?:\s?\w+){0,3}?)" + lookbehinds).replace("\w", "(?:" + lookaheads + "\w)")

        def find_regular_match(template_string, idx, action_to_use):

            output = [-1, [], []]

            match_obj = re.fullmatch(template_string.replace("OBJ", regex_string), action_to_use)

            if match_obj != None:
                output[0] = idx
                for i in range(0, len(match_obj.groups())):
                    word = match_obj.group(i + 1)
                    for individual_word in word.split(" "):
                        output[i + 1].append(self.vocab_act_rev[individual_word[0:min(6, len(individual_word))].lower()])
                return output
            return None

        def edit_action(test_string, replace_cmd=None):
            removed_quotes =  "".join(test_string.split("\""))
            if replace_cmd != None:
                return removed_quotes.replace(replace_cmd, "CMD")
            return removed_quotes


        output = [-1, [], []]

        if "," in action:
            action = action[action.index(",") + 1:].strip(" ")

        for i, template_string in enumerate(self.templates):

            if "CMD" in template_string:
                first_word = template_string.split(" ")[0]
                quote_regex = re.compile("(?:" + first_word + ".*?(\"[\w|\s]+\"))|(?:" + first_word + ".*?to ([\w|\s]+))")
                quote_match = re.search(quote_regex, action)

                if quote_match != None:

                    idx = 1 if quote_match.group(1) != None else 2
                    inner_command = quote_match.group(idx).lstrip("\"").rstrip("\"")
                    for j, inner_template in enumerate(self.templates):
                        if "CMD" not in inner_template:
                            inner_match = find_regular_match(inner_template, j, inner_command)
                            if inner_match != None:
                                replaced_match = find_regular_match(template_string, i, edit_action(action, inner_command))
                                if replaced_match != None:
                                    return [replaced_match[0], self.get_object_index(replaced_match[1]), self.get_object_index(replaced_match[2])]

            regular_output = find_regular_match(template_string, i, edit_action(action))
            if regular_output != None:
                return [regular_output[0], self.get_object_index(regular_output[1]), self.get_object_index(regular_output[2])]

        return output

    def get_object_index(self, obj_indices):
        if len(obj_indices) == 0:
            return None
        full_word = " ".join(self.vocab_act[idx] for idx in obj_indices)
        keyword = extract_object(full_word)
        if len(keyword) == 0:
            return None
        return self.vocab_act_rev[keyword]


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


