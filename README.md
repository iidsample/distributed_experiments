This documentation provides steps for creating the bottleneck while training.
## Setting up Wondershaper
1. Setup Wondershaper
```bash
git clone https://github.com/magnific0/wondershaper
cd wondershaper
sudo make install
```
2. Run ```ip addr``` or ```ifconfig```and find the network interface on which you want to reduce
   the bandwidth. In most cases on aws you will find this interface to be ens3.
3. Run the following commands to set up uplink limit ```1000 Kbps``` and
   downlink ```2000 Kbps``` on ens3 network interface
```bash
sudo ./wondershaper -a ens3 -u 1000 -d 2000
```
4. To clear the limit on a interface. Look at the following command. In the
   following command we clear the limits on ens3.
```bash
sudo ./wondershaper -c -a ens3
```
5. Also keep in mind that you need to clear limits before updating them.

## Running the pytorch code
Run the following code with right paremeters based on your setup.
 ```bash
python imagenet_example.py -a resnet50 --lr 0.01 --dist-url 'tcp://172.31.71.89:2345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 ~/data
```
