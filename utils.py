import torch
import heapq
import pickle
import gzip
import editdistance
import spacy
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def lora_to_base(model):
    model.base_model.disable_adapter_layers()
    model.eval()

def base_to_lora(model):
    model.base_model.enable_adapter_layers()
    model.train()

@torch.no_grad()
def score_fast(
    model,
    tokenizer,
    encoded_input,
    termination_token_id,
    min_len,
    skip_first,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    prompt_cache=None,
    reward_model=None,
    reward_tokenizer=None,
    classifier=None,
):
    batch_size = encoded_input.shape[0]
    seq_length = encoded_input.shape[1] - skip_first
    device = encoded_input.device
    
    decoded_texts = tokenizer.batch_decode(encoded_input, skip_special_tokens=True)
    # print("Decoded output after generation: \n",decoded_texts)
    reward_encoded = reward_tokenizer[0](
        decoded_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    if vocab_nice_mask is not None and not isinstance(vocab_nice_mask, torch.Tensor):
        vocab_nice_mask = torch.from_numpy(vocab_nice_mask)
    if vocab_naughty_mask is not None and not isinstance(vocab_naughty_mask, torch.Tensor):
        vocab_naughty_mask = torch.from_numpy(vocab_naughty_mask)
    
    reward = torch.zeros(batch_size, seq_length + 1).to(device)
    reward_unpenalized = torch.zeros(batch_size, seq_length + 1).to(device)
    
    non_term_mask = (encoded_input != termination_token_id)[:, skip_first:]
    non_term_mask = torch.cat(
        (
            non_term_mask.new_ones(non_term_mask.shape[0], 1),
            non_term_mask,
        ),
        dim=-1,
    )
    
    for i in range(seq_length + 1):
        prefix = reward_encoded['input_ids'][:, :skip_first + i]
        
        with torch.no_grad():
            rewards = [rm(prefix.to(reward_model.device)).logits[:, 0] for rm in reward_model]
            current_reward = rewards[0]
            reward[:, i] = current_reward
            reward_unpenalized[:, i] = current_reward
    
    reward[~non_term_mask] = 0.0
    reward_unpenalized[~non_term_mask] = 0.0
    # print("Tokenwise Reward: ",reward)
    return reward, reward_unpenalized

class FrozenModelSentenceGivenPrompt:
    def __init__(
        self,
        sentence_token_id,
        temperature=1.0,
        min_len=1,
        vocab_alpha=-50.0,
        vocab_nice_mask=None,
        vocab_naughty_mask=None,
        sentence_validator=None,
        valid_sentence_alpha=None,
        reward_model=None,
        reward_tokenizer=None,
        classifier=None,
    ):
        assert (
            sentence_validator is None
            and valid_sentence_alpha is None
            or sentence_validator is not None
            and valid_sentence_alpha is not None
        )

        self.temperature = temperature
        self.sentence_token_id = sentence_token_id
        self.vocab_nice_mask = vocab_nice_mask
        self.vocab_naughty_mask = vocab_naughty_mask
        self.vocab_alpha = vocab_alpha
        self.min_len = min_len
        self.sentence_validator = sentence_validator
        self.valid_sentence_alpha = valid_sentence_alpha
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.classifier = classifier

    def score(self, input_batch, prompt_length, model, tokenizer ,reward_model, reward_tokenizer,classifier):
        # lora_to_base(model)
        training = model.training
        model.eval()
        reward, reward_unpenalized = score_fast(
            model=model,
            tokenizer=tokenizer,
            encoded_input=input_batch,
            termination_token_id=self.sentence_token_id,
            skip_first=prompt_length,
            vocab_nice_mask=self.vocab_nice_mask,
            vocab_naughty_mask=self.vocab_naughty_mask,
            vocab_alpha=self.vocab_alpha,
            min_len=self.min_len,
            reward_model = reward_model,
            reward_tokenizer = reward_tokenizer,
            classifier=classifier
        )
        reward /= self.temperature
        reward_unpenalized /= self.temperature
        # base_to_lora(model)
        if training:
            model.train()

        if self.sentence_validator is not None:
            invalid = self.sentence_validator(input_batch[:, prompt_length:], tokenizer)
            invalid = invalid * self.valid_sentence_alpha
            reward = torch.min(reward, invalid)

        return reward, reward_unpenalized

class SentenceValidator:
    def __init__(self, sentence_token_id) -> None:
        self.sentence_token_id = sentence_token_id

    def __call__(self, sentences, tokenizer):
        pass

class RuleSentenceValidator(SentenceValidator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("en_core_web_lg")

    def __call__(self, sentences, tokenizer):
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=sentences.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        for i in range(sentences.shape[0]):
            for j in range(sentences.shape[1]):
                if sentences[i, j] == self.sentence_token_id:
                    break  # Only unterminated sentences get a reward
                sent = tokenizer.decode(sentences[i, : j + 1])
                sent = self.nlp(sent).sents
                tokens = []
                for s in sent:
                    for t in s:
                        tokens.append(t)
                if not (len(tokens) >= 2 and tokens[0].is_space and tokens[1].is_title):
                    invalid[i, j + 1] = True  # Must start with a space and capital
                    continue
                has_noun = 1
                has_verb = 1
                for token in tokens:
                    if token.pos_ in ["NOUN", "PROPN", "PRON"]:
                        has_noun -= 1
                    elif token.pos_ in ["VERB", "AUX"]:
                        has_verb -= 1
                if has_noun > 0 or has_verb > 0:
                    invalid[i, j + 1] = True  # Must have a noun and a verb
        return invalid

class ModelSentenceValidator(SentenceValidator):
    def __init__(self, *args, model_name=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if model_name is None:
            model_name = "textattack/roberta-base-CoLA"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )

    @torch.no_grad()
    def __call__(self, sentences, tokenizer):
        sentences = sentences.to(self.model.device)
        invalid = torch.zeros(
            sentences.shape[0],
            sentences.shape[1] + 1,
            dtype=torch.bool,
            device=self.model.device,
        )
        invalid[:, 0] = True  # Empty sentence is never valid
        done = torch.zeros(sentences.shape[0]).bool().to(self.model.device)
        for i in range(sentences.shape[1]):
            sent = sentences[:, : i + 1]
            done |= sent[:, -1] == self.sentence_token_id
            if done.all():
                break
            sent = self.tokenizer(
                tokenizer.batch_decode(sent),
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
            invalid_probs = self.model(**sent).logits.softmax(dim=-1)[:, 0]
            invalid[~done, i + 1] = invalid_probs[~done] > 0.2
        return invalid

def generate_and_return_termination_logprob(
    model,
    encoded_prompt,
    termination_token_id,
    reward_fn,
    vocab_nice_mask=None,
    vocab_naughty_mask=None,
    vocab_alpha=-99,
    max_len=10,
    min_len=0,
    temperature=1.0,
    top_k=999999,
    top_p=1.0,
    action_seq=None,
    skip_rewards=False,
):
    # print("Input Prompt Encoded: \n",encoded_prompt)
    # generate and return the probability of terminating at every step
    active_seqs = torch.ones(encoded_prompt.size(0)).bool().to(encoded_prompt.device)
    state = encoded_prompt.clone()
    action_seq = None
    # print("Encoded Prompt: ",state)
    log_pf = []
    log_pterm = []
    token_ids = state  # For caching hidden states during generation
    past_key_values = None  # For caching hidden states during generation
    for i in range(max_len + 1):
        output = model(input_ids=token_ids, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]
        if action_seq is None:
            with torch.no_grad():
                prob = logits.softmax(dim=-1)
                modified_logits = logits.clone().detach()
                # implement top-k by getting the top-k largest values and setting the rest to 0
                if top_k < 999999:
                    modified_logits[prob >= prob.topk(top_k)] = -torch.inf
                # implement top-p by getting indices in the top-p prob mass and setting the rest to 0
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(prob, dim=-1, descending=True)
                    cumsum_prob = torch.cumsum(sorted_probs, dim=-1)

                    nucleus = cumsum_prob < top_p

                    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)

                    mask = torch.zeros_like(prob, dtype=torch.bool)
                    for idx in range(prob.size(0)):
                        mask[idx, sorted_indices[idx, nucleus[idx]]] = True
                    modified_logits[~mask] = -torch.inf
                
                if i < min_len:
                    # if we haven't reach the minimum length, set the probability of terminating to 0
                    modified_logits[:, termination_token_id] = -torch.inf
                elif i >= max_len:
                    # if we've reached the maximum length, set the probability of terminating to 1
                    mask = [True] * modified_logits.shape[1]
                    mask[termination_token_id] = False
                    modified_logits[:, mask] = -torch.inf
                if vocab_nice_mask is not None:
                    # add vocab_alpha to the logits of the unmasked vocab items
                    modified_logits[:, ~vocab_nice_mask] += vocab_alpha
                if vocab_naughty_mask is not None:
                    if isinstance(vocab_naughty_mask, np.ndarray):
                        vocab_naughty_mask = torch.tensor(vocab_naughty_mask, dtype=torch.bool)
                    desired_size = modified_logits.shape[1]
                    current_size = len(vocab_naughty_mask)
                    if current_size < desired_size:
                        padding = torch.zeros(desired_size - current_size, dtype=torch.bool, device=vocab_naughty_mask.device)
                        vocab_naughty_mask = torch.cat((vocab_naughty_mask, padding))
                    elif current_size > desired_size:
                        vocab_naughty_mask = vocab_naughty_mask[:desired_size]
                    modified_logits[:, vocab_naughty_mask] += vocab_alpha
                # prob = (modified_logits / temperature).softmax(dim=-1)
                prob = (modified_logits / 0.5).softmax(dim=-1)

                token_ids = torch.multinomial(prob, num_samples=1)
                # token_ids = torch.argmax(prob, dim=-1, keepdim=True)

        else:
            if i >= action_seq.size(-1):
                token_ids = (
                    torch.ones_like(action_seq[:, 0]) * termination_token_id
                ).unsqueeze(-1)
            else:
                token_ids = action_seq[:, i].unsqueeze(-1)
        token_ids = torch.where(
            active_seqs.unsqueeze(-1),
            token_ids,
            termination_token_id,
        )
        if vocab_nice_mask is not None:
            logits[:, ~vocab_nice_mask] += vocab_alpha
        if vocab_naughty_mask is not None:
            logits[:, vocab_naughty_mask] += vocab_alpha
        logprob = logits.log_softmax(dim=-1)
        log_pterm.append(
            torch.where(
                active_seqs,
                logprob[:, termination_token_id],
                0,
            )
        )
        active_seqs = active_seqs * (token_ids != termination_token_id).squeeze(-1)
        log_pf.append(
            torch.where(
                active_seqs,
                logprob.gather(-1, token_ids).squeeze(-1),
                0,
            )
        )
        state = torch.cat([state, token_ids], dim=-1)
        # check if all sequences have terminated
        if torch.all(~active_seqs):
            break

    # print("Encoded output after generation \n",state)
    log_pf = torch.stack(log_pf, dim=1)
    log_pterm = torch.stack(log_pterm, dim=1)
    
    if skip_rewards:
        log_r, log_r_unpenalized = None, None
    else:
        # Reward for all intermediate states (except the last one,
        # which is guaranteed to be the termination token)
        log_r, log_r_unpenalized = reward_fn(state[:, :-1])
    
    # print("Log Prob of termination at each token: \n",log_pterm)
    # print("Log Prob of continuing at each token: \n",log_pf)
    # print(log_pf)
    # print(log_pterm)
    # add a termination token to the end of the sequence
    # print("log_pf:",log_pf)
    # print("log_pterm:",log_pterm)
    return state, log_pf, log_pterm, log_r, log_r_unpenalized


def modified_subtb_loss(
    log_pf,
    log_r,
    log_pterm,
    generated_text,
    termination_token_id,
    prompt_len,
    subtb_lambda=1.0,
):
    assert (
        log_pf.shape[1]
        == log_r.shape[1]
        == log_pterm.shape[1]
        == generated_text.shape[1] - prompt_len
    )
    assert (
        log_pf.shape[1] > 1
    )  # With modified-style losses, we need at least one transition before terminating

    delta = (
        log_r[:, :-1]
        + log_pf[:, :-1]
        + log_pterm[:, 1:]
        - log_r[:, 1:]
        - log_pterm[:, :-1]
    )
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)

    # Get a mask for tokens after the termination token in the generated_text
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = 0.0
    total_lambda = 0.0
    generated_len = generated_text.shape[1] - prompt_len
    for subtraj_len in range(1, generated_len):
        subtb_term = (
            delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]
        ) ** 2
        subtb_term[mask[:, subtraj_len - 1 :]] = 0
        batch_loss += subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
        total_lambda += (
            subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    batch_loss /= total_lambda
    # print("SubTrajLoss->Input(log_pf,log_r,log_pterm): ",batch_loss)

    return batch_loss


def get_termination_vals(
    generated_text,
    log_pf,
    log_pterm,
    log_r,
    log_r_unpenalized,
    termination_token_id,
    prompt_len,
):
    batch_idx = torch.arange(generated_text.size(0))
    gen_len = (
        (generated_text[:, prompt_len:] == termination_token_id).byte().argmax(dim=-1)
    )
    if log_pf is None and log_pterm is None:
        log_pfs = None
    else:
        log_pf = torch.cat([torch.zeros_like(log_pf[:, :1]), log_pf], dim=-1)[:, :-1]
        log_pfs = log_pf.cumsum(dim=-1) + log_pterm
        log_pfs = log_pfs[batch_idx, gen_len]
    log_r = log_r[batch_idx, gen_len]
    log_r_unpenalized = log_r_unpenalized[batch_idx, gen_len]
    return log_pfs, log_r, log_r_unpenalized, gen_len


class SequenceDiversity:
    def __init__(self, method, **kwargs):
        self.method = method
        if method is None:
            pass
        elif method == "sequence_embedding":
            model_name = kwargs.get(
                "model_name", "sentence-transformers/all-mpnet-base-v2"
            )
            self.model = SentenceTransformer(model_name)
        else:
            raise ValueError(f"Unknown sequence diversity method: {method}")

    @torch.no_grad()
    def __call__(self, sequences):
        if self.method is None:
            return None
        elif self.method == "sequence_embedding":
            embeddings = self.model.encode(sequences, show_progress_bar=False)
            sim = cos_sim(embeddings, embeddings)
            indices = torch.triu_indices(len(sequences), len(sequences), offset=1)
            diversity = 1 - sim[indices[0], indices[1]].mean().item()
        else:
            raise ValueError(f"Unknown sequence diversity method: {self.method}")
        return diversity


class ReplayBuffer:
    """
    A relay buffer that uses a heap to keep the max_size items with the highest reward
    """

    def __init__(self, buffer_size, termination_token_id, sim_tolerance=0.25):
        self.buffer_size = buffer_size
        self.termination_token_id = termination_token_id
        self.sim_tolerance = sim_tolerance
        self.reset()

    def reset(self):
        self._buffer = {}

    def add(self, item):
        """
        add an item to the buffer, where item = [log reward, tensor of shape (seq_len, )]
        """
        # if item is already in the buffer, skip it
        str_prompt = item["str_prompt"]
        if item["str_sentence"] in self._buffer[str_prompt]["exists"]:
            return
        # if the edit distance between item and any item in the buffer is small, skip it
        tokenized_sentence = [
            x
            for x in item["tensor_sentence"].tolist()
            if x != self.termination_token_id
        ]
        for buffer_item in self._buffer[str_prompt]["sentences"]:
            tokenized_existing_sentence = [
                x for x in buffer_item[2].tolist() if x != self.termination_token_id
            ]
            if (
                editdistance.eval(tokenized_sentence, tokenized_existing_sentence)
                < (len(tokenized_sentence) + len(tokenized_existing_sentence))
                * self.sim_tolerance
            ):
                if buffer_item[0] >= item["logreward"]:
                    return
                else:
                    self._buffer[str_prompt]["exists"].remove(buffer_item[1])
                    self._buffer[str_prompt]["sentences"].remove(buffer_item)
                    heapq.heapify(self._buffer[str_prompt]["sentences"])
                    self._buffer[str_prompt]["exists"].add(item["str_sentence"])
                    heapq.heappush(
                        self._buffer[str_prompt]["sentences"],
                        (
                            item["logreward"],
                            item["str_sentence"],
                            item["tensor_sentence"],
                            item["full_logrewards"],
                        ),
                    )
                    return
        self._buffer[str_prompt]["exists"].add(item["str_sentence"])
        if len(self._buffer[str_prompt]["sentences"]) >= self.buffer_size:
            popped = heapq.heappushpop(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )
            self._buffer[str_prompt]["exists"].remove(popped[1])
        else:
            heapq.heappush(
                self._buffer[str_prompt]["sentences"],
                (
                    item["logreward"],
                    item["str_sentence"],
                    item["tensor_sentence"],
                    item["full_logrewards"],
                ),
            )

    def add_batch(self, prompt, sentences, logrewards, tokenizer):
        """
        add a batch of items to the buffer
        """
        str_prompt = " ".join([str(x) for x in prompt.tolist()])
        if str_prompt not in self._buffer:
            self._buffer[str_prompt] = {
                "tensor_prompt": prompt,
                "sentences": [],
                "exists": set(),
            }
        sentences[
            (sentences == self.termination_token_id).cumsum(dim=-1) >= 1
        ] = self.termination_token_id
        token_sentences = tokenizer.batch_decode(sentences)
        for i in range(sentences.size(0)):
            str_sentence = token_sentences[i].replace(".", "").strip()
            self.add(
                {
                    "logreward": logrewards[
                        i, (sentences[i] != self.termination_token_id).sum()
                    ].item(),
                    "str_prompt": str_prompt,
                    "str_sentence": str_sentence,
                    "tensor_sentence": sentences[i],
                    "full_logrewards": logrewards[i, :],
                }
            )

    def sample(self, batch_size, prompt):
        """
        uniformly sample a batch of items from the buffer,
        and return a stacked tensor
        """
        str_prompt = " ".join([str(x) for x in prompt.tolist()])
        if str_prompt not in self._buffer:
            return None, None
        prompt_buffer = self._buffer[str_prompt]["sentences"]
        idx = np.random.choice(
            len(prompt_buffer),
            batch_size,
            replace=True,
        )
        return torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][2] for i in idx],
            batch_first=True,
            padding_value=self.termination_token_id,
        ), torch.nn.utils.rnn.pad_sequence(
            [prompt_buffer[i][3] for i in idx],
            batch_first=True,
            padding_value=0,
        )

    def print(self):
        for key in self._buffer:
            print(key)
            for item in self._buffer[key]["sentences"]:
                print(item[1])
            print("")

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self._buffer, f)
