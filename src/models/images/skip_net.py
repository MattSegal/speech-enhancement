"""
Skip net from Deep Image Prior: encoder-decoder with skip connections.
https://github.com/DmitryUlyanov/deep-image-prior
"""

num_input_channels = 32  # ???
num_output_channels = 3
num_channels_down = [128, 128, 128, 128]
num_channels_up = [128, 128, 128, 128]
num_channels_skip = [4, 4, 4, 4, 4]
filter_size_down = 3
filter_size_up = 3
filter_skip_size = 1
need_sigmoid = True
need_bias = True
pad = "reflect"
upsample_mode = "bilinear"
downsample_mode = "stride"
act_fun = "LeakyReLU"
need1x1_up = True

n_scales = 5
upsample_mode = [upsample_mode] * n_scales
downsample_mode = [downsample_mode] * n_scales
filter_size_down = [filter_size_down] * n_scales
filter_size_up = [filter_size_up] * n_scales
last_scale = n_scales - 1
cur_depth = None

model = nn.Sequential()
model_tmp = model


input_depth = num_input_channels
for i in range(n_scales):
    skip = nn.Sequential()
    deeper = nn.Sequential()
    deeper_main = nn.Sequential()
    # Add shit to model tmp
    model_tmp.add(Concat(1, skip, deeper))
    model_tmp.add(
        nn.BatchNorm2d(
            num_channels_skip[i]
            + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
        )
    )

    # Add shit to tmp
    skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
    skip.add(nn.BatchNorm2d(num_channels_skip[i]))
    skip.add(nn.LeakyReLU())

    # Add shit to deeper
    deeper.add(
        conv(
            input_depth,
            num_channels_down[i],
            filter_size_down[i],
            2,
            bias=need_bias,
            pad=pad,
            downsample_mode=downsample_mode[i],
        )
    )
    deeper.add(nn.BatchNorm2d(num_channels_down[i]))
    deeper.add(nn.LeakyReLU())
    deeper.add(
        conv(
            num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad
        )
    )
    deeper.add(nn.BatchNorm2d(num_channels_down[i]))
    deeper.add(nn.LeakyReLU())
    if i == len(num_channels_down) - 1:
        # The deepest
        k = num_channels_down[i]
    else:
        deeper.add(deeper_main)
        k = num_channels_up[i + 1]
    deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

    # Add more shit to model_tmp
    model_tmp.add(
        conv(
            num_channels_skip[i] + k,
            num_channels_up[i],
            filter_size_up[i],
            1,
            bias=need_bias,
            pad=pad,
        )
    )
    model_tmp.add(nn.BatchNorm2d(num_channels_up[i]))
    model_tmp.add(nn.LeakyReLU())
    if need1x1_up:
        model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(nn.BatchNorm2d(num_channels_up[i]))
        model_tmp.add(nn.LeakyReLU())

    input_depth = num_channels_down[i]
    model_tmp = deeper_main

model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
if need_sigmoid:
    model.add(nn.Sigmoid())


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
            np.array(inputs_shapes3) == min(inputs_shapes3)
        ):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2 : diff2 + target_shape2, diff3 : diff3 + target_shape3]
                )

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero", downsample_mode="stride"):
    downsampler = None
    if stride != 1 and downsample_mode != "stride":

        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ["lanczos2", "lanczos3"]:
            downsampler = Downsampler(
                n_planes=out_f,
                factor=stride,
                kernel_type=downsample_mode,
                phase=0.5,
                preserve_size=True,
            )
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

NUM_LAYERS = 5

class SkipNet(nn.Module):

    def __init__(self):
        super()__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skips = nn.ModuleList()
        # Build encoders
        for idx in range(NUM_LAYERS):
            in_channels = i * NUM_C
            out_channels = (i + 1) * NUM_C
            layer = nn.Sequential(*[
                nn.Conv2d(),
                nn.BatchNorm2d(),
                nn.LeakyReLU(),
            ])
