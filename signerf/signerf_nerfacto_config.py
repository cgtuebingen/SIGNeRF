"""" Configuration file for the SIGNeRF method combined with Nerfacto """

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig

from signerf.data.signerf_datamanager import SIGNeRFDataManagerConfig
from signerf.signerf_pipeline import SIGNeRFPipelineConfig
from signerf.signerf_trainer import SIGNeRFTrainerConfig
from signerf.data.signerf_dataparser import SIGNeRFDataParserConfig
from signerf.datasetgenerator.datasetgenerator import DatasetGeneratorConfig
from signerf.renderer.renderer import RendererConfig
from signerf.diffuser.diffuser import DiffuserConfig

signerf_nerfacto_method = MethodSpecification(
    config=SIGNeRFTrainerConfig(
        method_name="signerf_nerfacto",
        steps_per_save=1000,
        max_num_iterations=30000,
        save_only_latest_checkpoint=False,
        mixed_precision=True,
        pipeline=SIGNeRFPipelineConfig(
            datamanager=SIGNeRFDataManagerConfig(
                dataparser=SIGNeRFDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=NerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
            ),
            dataset_generator=DatasetGeneratorConfig(
                renderer=RendererConfig(),
                diffuser=DiffuserConfig(
                    url="http://127.0.0.1",
                    port=5000,
                    stable_diffusion_model="sd_xl_base_1.0.safetensors [31e35c80fc]", # Specify the stable diffusion model here
                    controlnet_model="diffusers_xl_depth_full [2f51180b]", # Specify the controlnet model here
                ),
            ), # Specify the dataset generator config here
        ),
        optimizers={
             "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-15, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=7007),
        vis="viewer",
    ),
    description="SIGNeRF method combined with Nerfacto (faster training less quality)",
)