from PIL import Image

from torch.utils.data import Dataset


class PlainDataset(Dataset):
    r"""
    Dataset using memcached to read data.

    Arguments
        * root (string): Root directory of the Dataset.
        * meta_file (string): The meta file of the Dataset. Each line has a image path
          and a label. Eg: ``nm091234/image_56.jpg 18``.
        * transform (callable, optional): A function that transforms the given PIL image
          and returns a transformed image.
    """

    def __init__(self, root, meta_file, transform=None):
        self.root = root
        self.transform = transform
        with open(meta_file) as f:
            meta_list = f.readlines()
        self.num = len(meta_list)
        self.metas = []
        for line in meta_list:
            path, cls = line.strip().split()
            self.metas.append((path, int(cls)))

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        filename = self.root + '/' + self.metas[index][0]
        cls = self.metas[index][1]

        with Image.open(filename) as img:
            img = img.convert('RGB')

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls
