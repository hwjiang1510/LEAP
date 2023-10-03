---

layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>

<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>LEAP</title>



<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->

<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>
<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/leap.png">
<link rel="icon" type="image/png" sizes="32x32" href="/leap.png">
<link rel="icon" type="image/png" sizes="16x16" href="/leap.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/leap.svg" color="#5bbad5">

<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->
<link rel="shortcut icon" type="image/x-icon" href="leap.ico">
</head>



<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong><br>LEAP: Liberate Sparse-view 3D Modeling <br /> from Camera Poses</strong></h1></center>
<center><h2>
    <a href="https://hwjiang1510.github.io/">Hanwen Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://zhenyujiang.me/">Zhenyu Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://zhaoyue-zephyrus.github.io/">Yue Zhao</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://www.cs.utexas.edu/~huangqx/">Qixing Huang</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="https://arxiv.org/pdf/2310.01410.pdf">Paper</a> | <a href="https://github.com/hwjiang1510/LEAP">Code</a> </h2></center>
<br>



<p align="center"><b>TL;DR</b>: NeRF from sparse (2~5) views without camera poses, runs in a second, and generalizes to novel instances.</p>
<br>

<h1 align="center">Real-World Demo</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./video/real_demo2.mp4"  type="video/mp4">
  </video>
  </td>
      </tr></tbody></table>
<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
    Are camera poses necessary for multi-view 3D modeling? Existing approaches predominantly assume access to accurate camera poses. While this assumption might hold for dense views, accurately estimating camera poses for sparse views is often elusive. Our analysis reveals that noisy estimated poses lead to degenerated performance for existing sparse-view 3D modeling methods. To address this issue, we present LEAP, a novel <b>pose-free</b> approach, therefore challenging the prevailing notion that camera poses are indispensable. LEAP discards pose-based operations and learns geometric knowledge from data. LEAP is equipped with a neural volume, which is shared across scenes and is parameterized to encode geometry and texture priors. For each incoming scene, we update the neural volume by aggregating 2D image features in a feature-similarity-driven manner. The updated neural volume is decoded into the radiance field, enabling novel view synthesis from any viewpoint. On both object-centric and scene-level datasets, we show that LEAP significantly outperforms prior methods when they employ predicted poses from state-of-the-art pose estimators. Notably, LEAP performs on par with prior approaches that use ground-truth poses while running 400x faster than PixelNeRF. We show LEAP generalizes to novel object categories and scenes, and learns knowledge closely resembles epipolar geometry.
</p></td></tr></table>
</p>
  </div>
</p>

<br>

<hr>
<h1 align="center">Overview</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <a href="./src/overview.png"> <img
		src="./src/overview.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>
<table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
     (Left) Prior works use poses-based operations, i.e., projection, to map 2D image information into the 3D domain. However, under inaccurate poses, the 2D-3D association will be wrong, leading to incorrect 3D features and degenerated performance. (Right) In contrast, LEAP uses attention to assign weights to all 2D pixels adaptively. The operation is not reliant on camera poses, enabling LEAP directly perform inference on unposed images. To initialize the features of 3D points, LEAP introduces a parametrized neural volume, which is shared across all scenes. The neural volume is trained to encode geometry and texture priors. For each incoming scene, the neural volume gets updated by querying the 2D image features and decodes the radiance field. For each 3D query point on a casting ray, its features are interpolated from its nearby voxels.
</p></td></tr></table>
<br><br>


<hr>


<h1 align="center">Reconstruction Results</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <video muted autoplay loop width="100%">
      <source src="./video/results_omniobject3d_small.mp4"  type="video/mp4">
  </video>
  </td></tr></tbody></table>

<br>



<h2 align="center">Compare with Prior Arts</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <p align="justify" width="20%">LEAP outperforms prior works that use state-of-the-art pose estimators, as well as pose-free SRT and single-view-based Zero123. LEAP performs on par with prior works that use ground-truth poses.</p>
      <a href="./src/vis_compare_main.png"> <img
		src="./src/vis_compare_main.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>

<br>

<h2 align="center">Transfer Knowledge to Scene-level Dataset</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <p align="justify" width="20%">As a pose-free method that learns geometric knowledge from data, LEAP generally requires larger training data. However, we find a pre-trained LEAP on object-centric dataset can transfer to scene-level, with only tens of training sample. This implies that LEAP learns general geometric knowledge. The performance on the DTU dataset is comparable to SPARF, which requires dense correspondence as inputs and per-scene training of 1 day.</p>
      <a href="./src/vis_compare_dtu.png"> <img
		src="./src/vis_compare_dtu.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>

<br>

<hr>

<h1 align="center">Interpret LEAP</h1>
<h2 align="center">Learned Geometric Knowledge</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <p align="justify" width="20%">We input images of a small dot (in orange boxes), and the visualization of the reconstructed neural volume shows consistency with the epipolar lines of the small dot on target views. This implies LEAP mapps a 2D point as its 3D reprojection ray segment even though there are no reprojection operations.</p>
      <a href="./src/vis_geometry.png"> <img
		src="./src/vis_geometry.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>

<br>

<h2 align="center">2D-2D Attention</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <p align="justify" width="20%">The query pixel (shown in red) of the canonical view attends to the corresponding regions in canonical views.</p>
      <a href="./src/attn_animated.gif"> <img
		src="./src/attn_animated.gif" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>

<h2 align="center">Neural Volume</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <p align="justify" width="20%">We analyze the neural volume by slicing it and using PCA for visualization. The learned neural volume encodes the mean shape in high-dimensional space. After aggregating information from 2D images, it encodes the surface of the object in a coarse-to-fine manner. </p>
      <a href="./src/vis_emb.png"> <img
		src="./src/vis_emb.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>

<br>

<h2 align="center">3D-2D Attention</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <p align="justify" width="20%">We visualize the 3D-2D attention weights for selected voxels. We show that on-surface voxels attend to specific 2D regions and the attention demonstrates to be smooth on neighbour on-surface voxels. The attention of non-surface voxels diffuses. </p>
      <a href="./src/vis_attn3d.png"> <img
		src="./src/vis_attn3d.png" style="width:100%;"> </a>
    </td>
  </tr>
  </tbody>
</table>

<br>

<hr>

<h2 align="center">Related Project and Acknowledgement</h2>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
        <p align="justify" width="20%"><b> <a href="https://ut-austin-rpl.github.io/FORGE/">FORGE</a></b>: Sparse-view reconstruction by leveraging the syngergy between shape and pose. </p>
        <p align="justify" width="20%"><b> <a href="https://github.com/bradyz/cross_view_transformers/">Cross-view Transformers</a></b>: A cross-view transformer that maps unposed images (while with fixed poses) to a representation in another domain. </p>
        <p align="justify" width="20%">We thank Brady Zhou for his inspiring cross-view transformers, and we thank Shuhan Tan for proof-reading the paper. </p>
    </td>
  </tr>
  </tbody>
</table>

<hr>
<!-- <table align=center width=800px> <tr> <td> <left> -->
<center><h1>Citation</h1></center>
<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@article{jiang2022LEAP,
   title={LEAP: Liberate Sparse-view 3D Modeling from Camera Poses},
   author={Jiang, Hanwen and Jiang, Zhenyu and Zhao, Yue and Huang, Qixing},
   journal={ArXiv},
   year={2023},
   volume={2310.01410}
}
</code></pre>
</left></td></tr></table>




<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>

<center><h1>Acknowledgements</h1></center> 
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->

