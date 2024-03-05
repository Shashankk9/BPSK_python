import numpy as np
import matplotlib.pyplot as plt

#transmitter side
#The transmitter modelled here is of a length 10000, where symbols are mapped as 1 and -1. 
# The symbol modulation also has a Gaussian noise of std dev=0.25 and mean=0. 
# 1. Sequence of random number as bits to be sent
def generate_rand_num(length):
    return np.random.randint(0, 2, length) 

# 2. Mapping of the generated random bits to random numbers to symbols 
# 0-> -1 and 1-> 1
def symbols_map(random_number):
    return 2*random_number-1

# 3. Adding of gaussian noise at trasmitter side
def noise_trans(symbols,std_dev):
    noise=np.random.normal(0,std_dev, symbols.shape)
    return noise+symbols

#Channel Section 
#(The channel modeled here is simple lossy channel and also adds noise with std_dev=1 and mean=0)
# The channel loss is sweeped from 1 dB to 10dB
def channel_loss_noise(sent_symbols,loss,std_dev_chn):
    noise_channel=np.random.normal(0,std_dev_chn, symbols.shape)
    return sent_symbols*loss+noise_channel

#Receiver section
# The receiver does a direct detection of the received signal from the noisy and lossy channel
# Here the detector has a detection efficiency of n
# The detection converts each detected value again to -1 and 1 which is converted back to 0 and 1
# Later on the values are compared to the initial generated random number and SNR is plotted

def direct_detection(received_symbols,det_eff):
    """Perform direct detection."""
    a=np.sign(det_eff*received_symbols) #signum function converts to -1 and 1 based on sign of received signal
    received_bit=(a+1)/2
    return received_bit

def calc_error(received_bit,random_number):
    "Calculate total sum of all the received bit which is not equal to initial random bit"
    return np.sum(received_bit != random_number)


#transmitter

bit_length=10000
std_dev_tra=0.1      #standard deviation of noise generated at transmitter

bit_sent=generate_rand_num(bit_length)
symbols=symbols_map(bit_sent)
sent_symbols=noise_trans(symbols,std_dev_tra)  #noisy symbols that is sent from transmitter. 

#Channel with loss and noise
std_dev_chn=1                         # standard deviation of channel's noise = 1
channel_loss_dB=np.arange(1,31,1)    #loss in dB
loss_factor=10**(-channel_loss_dB/10)

#det_eff=0.8

error_rate=np.zeros(len(loss_factor))

#sweeping for loss from 1 to 30dB
for i, loss in enumerate(loss_factor):
    # Channel
    print("loss:", loss)
    received_symbols = channel_loss_noise(sent_symbols,loss,std_dev_chn)

    #receiver part 
    # Receiver
    decoded_bits = direct_detection(received_symbols,det_eff=0.8)
    ber = calc_error(bit_sent, decoded_bits)
    error_rate[i] = ber / bit_length
    print("Bit Error Rate:", error_rate[i])


# Plot
#Bit error vs Loss plotted from 1 to 30dB
plt.figure(1)
plt.plot(channel_loss_dB, error_rate*100, marker='o')
plt.xlabel('Loss (dB)')
plt.ylabel('Bit Error Rate(%)')
plt.title('Bit Error Rate vs. Loss')
plt.grid(True)
plt.show()
    
