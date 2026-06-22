"""Dataset abstractions for contrastive pretraining."""


class ContrastiveDataset:
    """Minimal wrapper to return two augmented views and a label/index."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, target = self.samples[idx]
        if self.transform is None:
            return image, image, target
        view1, view2 = self.transform(image)
        return view1, view2, target
