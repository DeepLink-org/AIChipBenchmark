import torch
from torch import nn
from torch.nn import functional as F
from utils import fix_rand

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def permute_wrap(x):
    return x.permute((2, 0, 1))


def repeat_wrap(x):
    return x.repeat((2, 2, 1, 1))


def stack_wrap(x, y):
    return torch.stack((x, y), dim=1)


def cat_wrap(x, y):
    return torch.cat((x, y), dim=1)


def fill_wrap(x):
    return x.fill_(1.5)


def expand_wrap(x):
    return x.expand(-1, 4)


def scatter_add_wrap(x, index, src):
    return torch.scatter_add(x, dim=0, index=index, src=src)


def index_copy_wrap(x, index, source):
    return torch.index_copy(x, dim=0, index=index, source=source)


def index_put_wrap(x, values):
    return (
        torch.index_put(x, indices=(torch.tensor([0, 2]).to(device), torch.tensor([0, 2]).to(device)), values=values),
    )


def gen_unpool_inputs():
    pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    input = torch.rand(2, 3, 40, 50)
    output, indices = pool(input)
    output = output.detach().requires_grad_()
    indices = indices.detach().requires_grad_(False)
    return [output, indices]


def gen_rand(shape, grad=True):
    out = torch.rand(shape)
    out.requires_grad = grad
    return out


def gen_ctcloss_input():
    T = 50  # Input sequence length
    C = 20  # Number of classes (including blank)
    N = 16  # Batch size
    S = 30  # Target sequence length of longest target in batch (padding length)
    S_min = 10  # Minimum target length, for demonstration purposes
    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
    return [input, target, input_lengths, target_lengths]


def gen_bool(shape, grad=False):
    x = torch.randn(shape)
    y = x > 0
    y.requires_grad = grad
    return y


fix_rand()

configs = dict(
    conv2d=(
        nn.Conv2d(
            16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1), groups=1, bias=True, padding_mode="zeros"
        ),
        ((2, 16, 50, 40)),
    ),
    ConvTranspose2d=(nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2)), (2, 16, 50, 44)),
    batchnorm2d=(nn.BatchNorm2d(100, affine=False), (20, 100, 35, 45)),
    groupnorm=(nn.GroupNorm(3, 6), (20, 6, 10, 10)),
    layernorm=(nn.LayerNorm([3, 20, 24]), (2, 3, 20, 24)),
    instance_norm2d=(nn.InstanceNorm2d(100), (2, 100, 30, 40)),
    elu=(nn.ELU(), (2, 3, 100)),
    relu=(nn.ReLU(), (1, 3, 100, 100)),
    leaky_relu=(nn.LeakyReLU(0.1), (2, 3, 40, 200)),
    prelu=(nn.PReLU(), (2, 30, 100)),
    adaptive_avg_pool2d=(nn.AdaptiveAvgPool2d((5, 4)), (1, 64, 30, 8)),
    adaptive_max_pool2d=(nn.AdaptiveMaxPool2d((5, 4)), (1, 64, 30, 80)),
    avgpool2d=(nn.AvgPool2d((3, 2), stride=(2, 1)), (2, 10, 120, 40)),
    maxpool2d=(nn.MaxPool2d((3, 2), stride=(2, 1)), (2, 10, 12, 40)),
    max_unpool2d=(nn.MaxUnpool2d(2, stride=2), gen_unpool_inputs()),
    binary_cross_entropy=(F.binary_cross_entropy, [gen_rand((2, 100), True), torch.rand(2, 100)]),
    cross_entropy=(nn.CrossEntropyLoss(), [(3, 100), torch.ones(3).long()]),
    mse_loss=(nn.MSELoss(), [(3, 100), torch.randn(3, 100)]),
    kl_div=(nn.KLDivLoss(reduction="batchmean"), [(3, 100), (3, 100)]),
    l1_loss=(nn.L1Loss(), [(3, 100), (3, 100)]),
    ctc_loss=(nn.CTCLoss(), gen_ctcloss_input()),  # not for half
    nll_loss=(nn.NLLLoss(), [(1, 16, 100, 10), torch.ones(1, 100, 10).long()]),
    abs=(torch.abs, (2, 3, 50)),
    acos=(torch.acos, (32, 32)),
    asin=(torch.asin, (32, 32)),
    atan=(torch.atan, (32, 32)),
    ceil=(torch.ceil, (3, 32, 10)),
    cos=(torch.cos, (2, 3, 50)),
    cosh=(torch.cosh, (2, 3, 50)),
    erf=(torch.erf, (2, 3, 100)),
    erfc=(torch.erfc, (2, 3, 100)),
    erfinv=(torch.erfinv, (2, 3, 100)),
    exp=(torch.exp, (2, 50)),
    exp2=(torch.exp2, (2, 50, 10)),
    expm1=(torch.expm1, (2, 50)),
    floor=(torch.floor, (2, 3, 50)),
    fmod=(torch.fmod, (2, 3, 50), (2,)),
    log=(torch.log, (2, 3, 50)),
    log2=(torch.log2, (4, 2, 3, 50)),
    pow=(torch.pow, (2, 3, 50), (2,)),
    reciprocal=(torch.reciprocal, (3, 100)),
    round=(torch.round, (3, 100)),
    rsqrt=(torch.rsqrt, (3, 100)),
    sin=(torch.sin, (2, 3, 50)),
    sinh=(torch.sinh, (2, 3, 50)),
    sqrt=(torch.sqrt, (2, 3, 50)),
    tan=(torch.tan, (32, 32)),
    sigmoid=(torch.sigmoid, (2, 1000)),
    softmax=(nn.Softmax(dim=1), (2, 300)),
    log_softmax=(nn.LogSoftmax(dim=1), (2, 3, 100, 100)),
    vector_norm=(torch.linalg.norm, (2, 100), dict(dim=1)),
    matrix_norm=(torch.linalg.norm, (2, 100), dict(dim=(0, 1))),
    mode=(torch.mode, (2, 3, 100, 100)),
    maximum=(torch.maximum, [(2, 100), (2, 100)]),
    minimum=(torch.minimum, [(2, 100), (2, 100)]),
    eq=(torch.eq, [(20, 10), (10)]),
    ge=(torch.ge, [(2, 10), (10)]),
    gt=(torch.gt, [(2, 10), (2, 10)]),
    le=(torch.le, [(2, 10, 20), (2, 10, 20)]),
    lt=(torch.lt, [(2, 10), (2, 10)]),
    ne=(torch.ne, [(2, 10), (2, 10)]),
    logical_and=(torch.logical_and, [gen_rand((2, 100), False), gen_rand((2, 100), False)]),
    logical_or=(torch.logical_or, [gen_rand((2, 100), False), gen_rand((2, 100), False)]),
    logical_not=(torch.logical_not, gen_rand((2, 100), False)),
    logical_xor=(torch.logical_xor, [gen_rand((2, 100), False), gen_rand((2, 100), False)]),
    equal=(torch.equal, [(2, 10), (2, 10)]),
    add=(torch.add, [(2, 10), (10)]),
    sub=(torch.sub, [(2, 10, 300), (10, 300)]),
    div=(torch.div, [(2, 10, 30, 40), (10, 30, 40)]),
    mul=(torch.mul, [(2, 100), (2, 100)]),
    remainder=(torch.remainder, [(2, 100), gen_rand(100, False)]),
    fill_=(fill_wrap, gen_rand((2, 100), False)),
    where=(torch.where, [gen_bool((2, 10)), (2, 10), (2, 10)]),
    narrow=(torch.narrow, (2, 3), dict(dim=0, start=0, length=2)),
    split=(torch.split, (4, 10, 10), dict(split_size_or_sections=2)),
    flip=(torch.flip, (2, 3, 100), dict(dims=[0, 1])),
    transpose=(torch.transpose, (2, 33, 140), (0, 1)),
    expand=(expand_wrap, (10, 1)),
    repeat=(repeat_wrap, (1, 2, 100, 100)),
    stack=(stack_wrap, [(2, 3, 4), (2, 3, 4)]),
    cat=(cat_wrap, [(2, 3, 4), (2, 3, 4)]),
    fold=(nn.Fold(output_size=(21, 21), kernel_size=(2, 2)), (1, 400, 400)),
    unfold=(nn.Unfold(kernel_size=(2, 3)), (2, 5, 30, 100)),
    select=(torch.select, (2, 4), (1, 2)),
    scatter_add=(
        scatter_add_wrap,
        [(3, 5), torch.tensor([[0, 1, 2, 0, 0], [0, 1, 2, 2, 2]]).to(device), torch.ones((2, 5)).to(device)],
    ),
    index_copy=(index_copy_wrap, [(5, 3), torch.tensor([0, 4, 2]), (3, 3)]),
    index_fill=(torch.index_fill, (3, 3), dict(dim=1, index=torch.tensor([0, 2]).to(device), value=-1)),
    index_put=(index_put_wrap, [(3, 3), (2)]),
    masked_fill=(torch.masked_fill, [(2, 100), gen_bool((2, 100))], dict(value=1.0)),
    masked_scatter=(torch.masked_scatter, [(2, 100), gen_bool((2, 100)), (2, 100)]),
    masked_select=(torch.masked_select, [(2, 100), gen_bool((2, 100))]),
    dot=(torch.dot, [(100), (100)]),
    mv=(torch.mv, [(20, 40), (40)]),
    linear=(nn.Linear(20, 30), (128, 20)),
    bilinear=(nn.Bilinear(20, 30, 40), [(128, 20), (128, 30)]),
    mm=(torch.mm, [(20, 30), (30, 50)]),
    ger=(torch.ger, [(10,), (5)]),
    bmm=(torch.bmm, [(2, 3, 400), (2, 400, 60)]),
    matmul=(torch.matmul, [(10, 3, 4), (4, 5)]),
    addcdiv=(torch.addcdiv, [(120, 120), (120, 120), (120, 120)], dict(value=0.1)),
    addcmul=(torch.addcmul, [(10, 10), (10, 10), (10, 10)], dict(value=0.13)),
    addmv=(torch.addmv, [(2), (2, 20), (20)]),
    addmm=(torch.addmm, [(2, 10), (2, 20), (20, 10)]),
    corss=(torch.cross, [(10, 3), (10, 3)], dict(dim=1)),
    diagonal=(torch.diagonal, (10, 10), dict(offset=1)),
    trace=(torch.trace, (10, 10)),
    det=(torch.det, (2, 10, 10)),
    tril=(torch.tril, (10, 20)),
    triu=(torch.triu, (10, 20)),
    eig=(torch.eig, gen_rand((8, 8)), dict(eigenvectors=True), dict(backward=False)),
    svd=(torch.svd, (10, 20)),
    inverse=(torch.inverse, (10, 10)),
    interpolate=(F.interpolate, (2, 30, 40), dict(scale_factor=2)),
    grid_sample=(F.grid_sample, [(1, 3, 100, 100), (1, 50, 50, 2)], dict(align_corners=False)),
    affine_grid=(F.affine_grid, (4, 2, 3), dict(size=torch.Size((4, 3, 24, 24)), align_corners=False)),
    pad=(F.pad, (1, 3, 100, 100), ((1, 1), "constant", 0)),
    nonzero=(torch.nonzero, (1, 3, 20, 40), dict(as_tuple=False), dict(backward=False)),
    unique=(
        torch.unique,
        (2, 30, 40),
        dict(sorted=True, return_inverse=False, return_counts=False),
        dict(backward=False),
    ),
    topk=(torch.topk, (2, 1000), dict(k=3)),
    sort=(torch.sort, (2, 3, 100, 100)),
    threshold=(nn.Threshold(0.1, 20), (2, 3, 100)),
    clamp=(torch.clamp, (2, 3, 100, 100), dict(min=-0.5, max=0.5)),
    sign=(torch.sign, (2, 40)),
    all=(torch.all, torch.rand(4, 20).bool(), dict(dim=0)),
    any=(torch.any, torch.rand(4, 20).bool()),
    max=(torch.max, (2, 3, 100, 100)),
    min=(torch.min, (2, 3, 100, 100)),
    mean=(torch.mean, (2, 3, 100, 100)),
    sum=(torch.sum, (2, 3, 100, 100)),
    prod=(torch.prod, (2, 3, 40, 100)),
    median=(torch.median, (2, 3, 100, 100)),
    std=(torch.std, (2, 3, 100, 100), dict(dim=2, unbiased=True)),
    var=(torch.var, (2, 3, 100, 100), dict(dim=2, unbiased=True)),
)

non_deterministics = [
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "ctc_loss",
    "nll_loss",
    "scatter_add",
    "index_copys",
    "grid_sample",
]
