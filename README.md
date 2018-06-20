A simple implementation of seam carving, a content aware image resizing method.

Usage: python seam.py \<input image\> \<width\> \<height\> \<energy type\> \<output image\>
  
For energy type\\
0 = regular energy without entropy term\\
1 = regular energy with entropy term\\
2 = forward energy\\
3 = deep-based energy
  
For example:
<pre><code>python seam.py test/bird.jpeg 350 600 0 bird_rgb.jpeg
</code></pre>
for resizing bird.jpeg to 350×600 with RGB energy, and save to bird_rgb.jpeg

For help, see 
<pre><code>python seam.py --help
</code></pre>


