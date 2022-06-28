import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

# python generate.py --sample 4 --pics 4 --ckpt ./checkpoint/070000.pt

def generate(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            mask_img = sample[..., 3]
            fake_img_color = sample[..., :3]
            print(mask_img.shape, fake_img_color.shape)

            utils.save_image(
                fake_img_color.permute(0, 3, 1, 2),
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            utils.save_image(
                mask_img[..., None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                f"sample/{str(i).zfill(6)}_mask.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=4, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    # FIXME acquire base mesh, fix option acquire
    from process_data.mesh_handler import MeshHandler
    from process_data import options
    # (self, path_or_mesh: Union[str, T_Mesh], opt: Options, level: int, local_axes: Union[N, TS] = None)
    opt_ = options.Options()
    opt_.parse_cmdline(parser)
    template_mesh = MeshHandler(path_or_mesh="data/sphere.obj", opt=opt_)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, template_mesh, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
