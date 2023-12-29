# SIGNeRF
<p align="left">
  <strong>
    Scene Integrated Generation for Neural Radiance Fields
  </strong>
</p>



https://github.com/cgtuebingen/SIGNeRF/assets/9963865/edca0821-11da-482a-8695-138a2c022bbc




<p align="center">
    <span> 🌐  <a href="https://signerf.jdihlmann.com/"> Project Page </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📄  <a href="https://arxiv.org/"> Paper (Arxiv) </a> </span>&nbsp;&nbsp;&nbsp;
    <span> 📺  <a href="https://www.youtube.com/playlist?list=PL5y23CB9WmildtW3QyMEi3arXg06zB4ex"> Videos </a> </span>&nbsp;&nbsp;&nbsp;
</p>

# About
We propose SIGNeRF, a novel approach for fast and controllable NeRF scene editing and scene-integrated object generation. We introduce a new generative update strategy that ensures 3D consistency across the edited images, without requiring iterative optimization. We find that depth-conditioned diffusion models inherently possess the capability to generate 3D consistent views by requesting a grid of images instead of single views. Based on these insights, we introduce a multi-view reference sheet of modified images. Our method updates an image collection consistently based on the reference sheet and refines the original NeRF with the newly generated image set in one go. By exploiting the depth conditioning mechanism of the image diffusion model, we gain fine control over the spatial location of the edit and enforce shape guidance by a selected region or an external mesh.


# Code
Paper is currently under review, we will realse the code shortly in a cleaned version for [Nerfstudio](https://docs.nerf.studio/). Up until then if you have any questions regarding the project or need material to compare against fast, feel free to contact us. 


# Citation

You can find our paper on [arXiv](https://arxiv.org/).

If you find the paper useful for your research, please consider citing:

```
@inproceeding{signerf,
  author ={Dihlmann, Jan-Niklas and Engelhardt, Andreas and Lensch, Hendrik P.A.},
  title ={SIGNeRF: Scene Integrated Generation for Neural Radiance Fields},
  booktitle ={Preprint (ToDo)},
  year ={2023}
}
```
