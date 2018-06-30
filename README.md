A simple implementation of seam carving, a content aware image resizing method.

Usage: python seam.py \<input image\> \<width\> \<height\> \<energy type\> \<output image\>
  
For energy type  

* 0 = regular energy without entropy term  

* 1 = regular energy with entropy term  

* 2 = forward energy  

* 3 = deep-based energy

* 666 = deconvVGG19 energy
  
For example:
<pre><code>python seam.py test/bird.jpeg 350 600 0 bird_rgb.jpeg
</code></pre>
to resize bird.jpeg to 350×600 with RGB energy, and save to bird_rgb.jpeg

For help, see 
<pre><code>python seam.py --help
</code></pre>

To ensure that you have properly set up the environment for running, you can simple type the following command:
<pre><code>pip install -r requirements.txt
</code></pre>
or just run the script file to set up the environment as well as test the data we demonstrate in our report:
<pre><code>./run.sh
</code></pre>

If you prefer to set up the environment manually, please comment the second line in the script file like this:
<pre><code>#pip install -r requirements.txt
</code></pre>

