from torch.utils.data import Dataset


class EndToEndDataset(Dataset):
    def __init__(self, frames, questions, q_types, answers, lang_only=False):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    @staticmethod
    def from_baseline_dataset(dataset, lang_only=False):
        frames = []
        questions = []
        q_types = []
        answers = []
        for v_idx in range(len(dataset)):
            v_frames, v_qs, v_types, v_ans = dataset[v_idx]
            frames.append(v_frames)
            questions.append(v_qs)
            q_types.append(v_types)
            answers.append(v_ans)

        e2e_dataset = EndToEndDataset(frames, questions, q_types, answers, lang_only=lang_only)
        return e2e_dataset
