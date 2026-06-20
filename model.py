"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
import numpy as np
def argmax_rows(matrix):
    ans=[]
    for row in matrix:
        idx=0
        for i in range (1,len(row)):
            if row[i]>row[idx]:
                idx=i
        ans.append(idx)
    return np.array(ans)

# Step 2 - row_max
import numpy as np

def row_max(matrix):
   ans=[]
   for row in matrix:
    idx=0
    for i in range(1,len(row)):
      if row[i]>row[idx]:
         idx=i
    ans.append(row[idx])
   return np.array(ans).reshape(len(ans),1)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    ans=[]
    for row in matrix:
        sum=0
        for i in range(0,len(row)):
            sum+=row[i]
        ans.append(sum)
    return np.array(ans).reshape(len(ans),1)

# Step 4 - exp_shifted
import numpy as np
def exp_shifted(logits):
    ans=row_max(logits)
    logits-=ans
    result=[]
    for i in logits:
        result.append(np.exp(i))
    return np.array(result)

# Step 5 - stable_softmax
import numpy as np
def stable_softmax(logits):
    result=exp_shifted(logits)
    sum=row_sum(result)
    return result/sum

# Step 6 - one_hot
def one_hot(labels, num_classes):
    final=[]
    for label in labels :
        ans=[]
        for i in range(num_classes):
            if label ==i:
              ans.append(1)
            else:
               ans.append(0)
        final.append(ans)
    return np.array(final).astype(float)

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    ans = []

    for row_idx in range(len(labels)):
        label = labels[row_idx]
        ans.append(probs[row_idx][label])

    return np.array(ans)

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    ans=gather_true_class_probs(probs,labels)
    sum=0
    for i in ans:
        
        sum+= -np.log(i+eps)
    return sum/len(ans)

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    max=argmax_rows(logits_or_probs)
    sum=0
    for i in range(len(max)):
        if max[i]==labels[i]:
            sum+=1
    return sum/len(max)

# Step 10 - he_std
import numpy as np
def he_std(fan_in):
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    np.random.seed(seed)
    std=he_std(fan_in)
    w=np.random.randn(*shape)*std
    return w

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    w=np.zeros(length)
    return w

# Step 13 - pad_2d
def pad_2d(images, pad):
   return np.pad(images,(
    (0,0),(0,0),(pad,pad),(pad,pad)
   ))

# Step 14 - output_spatial_size
def output_spatial_size(le, kernel, stride, padding):
    out=le+2*padding-kernel
    out/=stride
    out+=1
    return int(out)

# Step 15 - im2col
import numpy as np

def im2col(images, kernel_h, kernel_w, stride, pad):
    N, C, H, W = images.shape

    padded = pad_2d(images, pad)

    out_h= output_spatial_size(
        H, 
        kernel_h,
        stride, pad
    )
    out_w= output_spatial_size(
        W, 
        kernel_w,
        stride, pad
    )

    cols = []

    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):

                h_start=oh*stride
                w_start=ow*stride

                patch = padded[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w
                ]

                cols.append(patch.reshape(-1))

    return np.array(cols, dtype=images.dtype)

# Step 16 - col2im
import numpy as np

def col2im(cols, input_shape, kernel_h, kernel_w, stride, pad):
    N, C, H, W = input_shape

    out_h = output_spatial_size(
        H,
        kernel_h,
        stride,
        pad
    )

    out_w = output_spatial_size(
        W,
        kernel_w,
        stride,
        pad
    )

   
    padded_h = H + 2 * pad
    padded_w = W + 2 * pad

    images = np.zeros(
        (N, C, padded_h, padded_w),
        dtype=cols.dtype
    )

    row = 0

    for n in range(N):
        for y in range(out_h):
            for x in range(out_w):
                patch = cols[row].reshape(
                    C,
                    kernel_h,
                    kernel_w
                )

                h_start = y * stride
                w_start = x * stride

                images[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w] += patch

                row += 1

    if pad > 0:
        return images[
            :,
            :,
            pad:-pad,
            pad:-pad
        ]

    return images

# Step 17 - conv2d_forward
import numpy as np
def im2col1(images, kernel_h, kernel_w, stride, pad):
    N, C, H, W = images.shape

    padded = pad_2d(images, pad)

    out_h= output_spatial_size(
        H, 
        kernel_h,
        stride, pad
    )
    out_w= output_spatial_size(
        W, 
        kernel_w,
        stride, pad
    )

    cols = []

    for n in range(N):
        for oh in range(out_h):
            for ow in range(out_w):

                h_start=oh*stride
                w_start=ow*stride

                patch = padded[
                    n,
                    :,
                    h_start:h_start + kernel_h,
                    w_start:w_start + kernel_w
                ]

                cols.append(patch.reshape(-1))

    return np.array(cols, dtype=images.dtype)
def conv2d_forward(x, weights, bias, stride, padding):
    N,C_in,H,W=x.shape
    C_out,C_in,kernel_h,kernel_w=weights.shape
    out_h = output_spatial_size(
    H,
    kernel_h,
    stride,
    padding)
    out_w = output_spatial_size(
    W,
    kernel_w,
    stride,
    padding)
    
    img=im2col1(x,kernel_h,kernel_w,stride,padding)
    ww=weights.reshape(
        C_out,-1
    ).T
    out=(img @ ww) + bias
    out = out.reshape(
    N,
    out_h,
    out_w,
    C_out
)
    out = out.transpose(0,3,1,2)
    dic={
        "x_shape":x.shape,
        "weights":weights,
        "cols":  img,
        "stride":stride,
        "padding":padding,
        "kernel_h":kernel_h,
        "kernel_w":kernel_w
    }
    return out,dic

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    N, C_out, out_h, out_w = d_out.shape 
    d_out=d_out.transpose(0,2,3,1)
    x_shape = cache["x_shape"]
    weights = cache["weights"]
    stride = cache["stride"]
    padding = cache["padding"]
    kernel_h = cache["kernel_h"]
    kernel_w = cache["kernel_w"]
    C_out,C_in,kh,kw=weights.shape
    ww=weights.reshape(C_out,-1)
    
    dd=d_out.reshape(-1,C_out)
    d_cols= dd @ ww
    ans=col2im(d_cols,x_shape,kernel_h,kernel_w,stride,padding)
    return np.array(ans)

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    weights = cache["weights"]
    cols=cache["cols"]
    C_out, C_in, kh, kw = weights.shape
    d_out=d_out.transpose(0,2,3,1)
    d_out=d_out.reshape(-1,C_out)#cout,n*oh,*ow
    ans=cols.T @ d_out
    ans=ans.T.reshape(
       C_out,C_in,kh,kw
    )
    return ans

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    return np.sum(d_out,axis=(0,2,3))

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):

   ip=conv2d_grad_input(d_out,cache)
   we=conv2d_grad_weights(d_out,cache)
   bb=conv2d_grad_bias(d_out)
   return ip,we,bb

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    pad=0
    N, C, H, W=x.shape
    out_h= output_spatial_size(
        H, 
        kernel,
        stride, pad
    )
    
    out_w= output_spatial_size(
        W, 
        kernel,
        stride, pad
    )
    aa=np.zeros((N,C,out_h,out_w))
    bb=np.zeros((N,C,out_h,out_w),dtype=np.int64)
    for n in range(N):
     for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                h=i*stride
                w=j*stride
                window=x[
                    n,c,h:h+kernel,w:w+kernel
                ]
                aa[n,c,i,j]=np.max(window)
                bb[n,c,i,j]=np.argmax(window)
    cache={
        "x_shape": x.shape,
        "argmax":bb,
        "kernel":kernel,
        "stride":stride

    }
    return  aa,cache

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    ans=np.zeros((kernel,kernel),dtype=np.float32)
    row=argmax_index//kernel
    column=argmax_index%kernel
    ans[row][column]=grad_value
    return ans

# Step 24 - maxpool2d_backward
import numpy as np
def maxpool2d_backward(d_out, cache):
    x_shape = cache["x_shape"]
    stride = cache["stride"]
    kernel = cache["kernel"]
    argmax=cache["argmax"]
    N, C, H, W=x_shape
    pad=0
    out_h= output_spatial_size(
        H, 
        kernel,
        stride, pad
    )
    result=np.zeros((N,C,H,W),dtype=np.float64)
    out_w= output_spatial_size(
        W, 
        kernel,
        stride, pad
    )
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    grad=d_out[n,c,i,j]
                    arg=argmax[n,c,i,j]
                    win=scatter_grad_window(grad,arg,kernel)
                    h=i*stride
                    w=j*stride
                    result[
                        n,c,h:h+kernel,w:w+kernel
                    ]+=win

    return result

# Step 25 - relu_forward
def relu_forward(x):
    cache = {"x": x}

    original_shape=x.shape

    x=x.reshape(-1)

    ans = []

    for v in x:
        ans.append(max(0, v))

    out = np.array(ans).reshape(original_shape).astype(np.float64)

    return out, cache

# Step 26 - relu_backward
def relu_backward(d_out, cache):
   x=cache["x"]
   
   return d_out*(x>0)

# Step 27 - flatten_forward
def flatten_forward(x):
    cache = {
        "x_shape": x.shape
    }
    N = x.shape[0]
    out = x.reshape(
        N,
        -1
    )
    return out, cache

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    x_shape=cache["x_shape"]
    return d_out.reshape(x_shape)

# Step 29 - linear_forward
def linear_forward(x, weights, bias):
    cache={
        "x":x,
        "weights":weights
    }
    ans=x @ weights +bias
    return ans,cache

# Step 30 - linear_grad_input
import numpy as np

def linear_grad_input(d_out, cache):
    weights=cache["weights"]
    return d_out@ weights.T

# Step 31 - linear_grad_weights
import numpy as np

def linear_grad_weights(x, dout):
   
   return x.T @ dout

# Step 32 - linear_grad_bias
import numpy as np

def linear_grad_bias(dout):
    return np.sum(dout,axis=0)

# Step 33 - linear_backward
def linear_backward(d,c):
    inp=linear_grad_input(d,c)
    x=c["x"]
    wi=linear_grad_weights(x,d)
    bb=linear_grad_bias(d)
    return inp,wi,bb

# Step 34 - softmax_cross_entropy_forward
def softmax_cross_entropy_forward(logits, y):
    sof=stable_softmax(logits)
    ans=cross_entropy_loss(sof,y,eps=0)
    return ans

# Step 35 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(logits, y):
    sof=stable_softmax(logits)
    N,C=logits.shape
    targets=one_hot(y,C)
    
    return (sof-targets)/N

# Step 36 - sgd_step
import numpy as np

def sgd_step(param, grad, lr):
    return param-grad*lr

# Step 37 - adam_update_m
import numpy as np

def adam_update_m(m, grad, beta_one):
    m*=beta_one
    m+=(1-beta_one)*grad
    return m

# Step 38 - adam_update_v
import numpy as np
def adam_update_v(v, grad, beta_two):
   v=beta_two*v +(1-beta_two)*(grad**2)
   return v

# Step 39 - adam_bias_correct
def adam_bias_correct(m, beta, t):
    d=(1-beta**t)
    return m/d

# Step 40 - adam_param_step
import numpy as np

def adam_param_step(param, m_hat, v_hat, lr, eps):
   d=np.sqrt(v_hat)+eps
   e= param-lr*m_hat/d
   return e

# Step 41 - adam_step
import numpy as np

def adam_step(param, grad, m, v, t, lr, beta_one, beta_two, eps):
    ne_be=adam_bias_correct(m,beta_one,t)
    
    new_v=adam_update_v(v,grad,beta_two)
    new_m=adam_update_m(m,grad,beta_one)
    m = adam_bias_correct(new_m, beta_one, t)
    v = adam_bias_correct(new_v, beta_two, t)
    new_param=adam_param_step(param,m,v,lr,eps)
    return new_param,new_m,new_v

# Step 42 - init_conv_layer
import numpy as np
def init_conv_layer(out_channels, in_channels, kernel_size, seed=0):
    b=np.zeros((out_channels,))
    np.random.seed(seed)
    shape=(out_channels, in_channels, kernel_size, kernel_size)
    in_channels*=kernel_size**2
    p=he_init(shape,in_channels,seed)
    dic={
        "W":p,"b":b
    }
    return dic

# Step 43 - init_linear_layer
def init_linear_layer(in_features, out_features, seed=0):
    np.random.seed(seed)
    b=init_zero_bias(out_features)
    w=he_init((in_features,out_features),in_features,seed)
    dic={
        "W":w,"b":b
    }
    return dic

# Step 44 - init_lenet
def init_lenet(i,num_classes, seed=0):
    params = {
        "conv1": init_conv_layer(6,i,5,0),
        "conv2": init_conv_layer(16,6,5,1),
        "fc1": init_linear_layer( 256 ,120,67),
        "fc2": init_linear_layer(120,num_classes,7)          
        
    }

    return params

# Step 45 - forward_conv_block
def forward_conv_block(x, w, b, kernel, stride, pad):
    out,dic=conv2d_forward(x,w,b,stride,pad)
    out,dic2=relu_forward(out)
    out,dic3=maxpool2d_forward(out, kernel,kernel)

    return out,{ "conv_cache":dic,"relu_cache":dic2,"pool_cache":dic3   }

# Step 46 - forward_classifier_block
def forward_classifier_block(x, fc1, fc2):
    flatten_out, flatten_cache = flatten_forward(x)
    fc1_out, fc1_cache = linear_forward(
        flatten_out,
        fc1["W"],
        fc1["b"])
    relu_out, relu_cache = relu_forward(fc1_out)
    logits, fc2_cache = linear_forward(
        relu_out,
        fc2["W"],
        fc2["b"]
    )
    cache = {
        "flatten_cache": flatten_cache,
        "fc1_cache": fc1_cache,
        "relu_cache": relu_cache,
        "fc2_cache": fc2_cache,
    }
    return logits, cache

# Step 47 - lenet_forward
def lenet_forward(x, params):
    out,block1=forward_conv_block(x,params["conv1"]["W"],params["conv1"]["b"],2,1,0)
    out,block2=forward_conv_block(out,params["conv2"]["W"],params["conv2"]["b"],2,1,0)
    out,classifier=forward_classifier_block(out,params["fc1"],params["fc2"])
    return out,{
        "block1":block1,"block2":block2,"classifier":classifier    }

# Step 48 - backward_conv_block
def backward_conv_block(dout, cache):
   out=maxpool2d_backward(dout,cache["pool_cache"])
   out=relu_backward(out,cache["relu_cache"])
   dx,dw,db=conv2d_backward(out,cache["conv_cache"])
   return dx,dw,db

# Step 49 - backward_classifier_block
def backward_classifier_block(dlogits, cache):
    dx1,dw1,db1=linear_backward(dlogits,cache["fc2_cache"])
    dx4=relu_backward(dx1,cache["relu_cache"])
    dx2,dw2,db2=linear_backward(dx4,cache["fc1_cache"])
    dx3=flatten_backward(dx2,cache["flatten_cache"])
    return {
        "dx":dx3,
        "fc1":{
            "dW":dw2,"db":db2

        },
        "fc2":{
            "dW":dw1,"db":db1
        }
    }

# Step 50 - lenet_backward
def lenet_backward(dlogits, cache):
    dic=backward_classifier_block(dlogits,cache["classifier"])
    dx2,dw2,db2=backward_conv_block(
        dic["dx"],cache["block2"]
    )
    dx1,dw1,db1=backward_conv_block(
        dx2,cache["block1"]
    )
    return {
    "conv1": {
        "dW": dw1,
        "db": db1
    },
    "conv2": {
        "dW": dw2,
        "db": db2
    },
    "fc1": {
        "dW": dic["fc1"]["dW"],
        "db": dic["fc1"]["db"]
    },
    "fc2": {
        "dW": dic["fc2"]["dW"],
        "db": dic["fc2"]["db"]
    }
}

# Step 51 - lenet_predict
def lenet_predict(x, params):
    N=x.shape[0]
    logits,dic=lenet_forward(x,params)
    return np.argmax(logits,axis=1).reshape(-1).astype(np.int64)

# Step 52 - build_synthetic_image_dataset
import numpy as np
def build_synthetic_image_dataset(num_samples, num_classes, image_size, in_channels=1, seed=0):
    rand=np.random.default_rng(seed)
    
    x=rand.integers(low=0,high=num_classes,size=num_samples)
    t=rand.standard_normal(size=(num_samples, in_channels, image_size,image_size))
    shift=x-(num_classes-1)/2
    t+=shift[:,None,None,None].astype(x.dtype)
    return t,x

# Step 53 - shuffle_indices
import numpy as np

def shuffle_indices(n, seed=0):
  
   np.random.seed(seed)
   
   
   return np.random.permutation(n)

# Step 54 - train_test_split
def train_test_split(x, y, test_fraction=0.2, seed=0):
    n=int(len(x)*(test_fraction))
    x_train=x[n:]
    x_test=x[:n]
    y_train=y[n:]
    x_test=shuffle_indices(x_test,seed)
    y_test=y[:n]
    y_test=shuffle_indices(y_test,seed)

    

    return x_train, y_train, x_test, y_test

# Step 55 - iterate_minibatches
def iterate_minibatches(x, y, batch_size, seed=0):
   shuf=shuffle_indices(len(x),seed)

   for i in range(0,len(y),batch_size):
      idx=shuf[i:i+batch_size]
      
      if (i+batch_size)>len(x): break
      ans =x[idx]
      ans1 =y[idx]

      yield ans,ans1

# Step 56 - train_step
def train_step(params, opt_state, xb, yb,
               lr, beta_one, beta_two,
               eps, step):

    logits, cache = lenet_forward(xb, params)
    loss = softmax_cross_entropy_forward(logits, yb)
    dlogits = softmax_cross_entropy_backward(logits, yb)
    grads = lenet_backward(dlogits, cache)

    new_params = {}
    new_opt_state = {}

    for layer in params:
        new_params[layer] = {}
        new_opt_state[layer] = {}

        for pname in params[layer]:
            param = params[layer][pname]
            grad = grads[layer]["d" + pname]   # "W"->"dW", "b"->"db"

            m = opt_state[layer][pname]["m"]
            v = opt_state[layer][pname]["v"]

            new_param, new_m, new_v = adam_step(
                param, grad, m, v, step,
                lr, beta_one, beta_two, eps
            )

            new_params[layer][pname] = new_param
            new_opt_state[layer][pname] = {"m": new_m, "v": new_v}

    return new_params, new_opt_state, loss

# Step 57 - train_one_epoch
def train_one_epoch(params, opt_state, x, y, batch_size,
                     lr, beta_one, beta_two, eps, step, seed=0):
    losses = []
    for xb, yb in iterate_minibatches(x, y, batch_size, seed):
        params, opt_state, loss = train_step(
            params, opt_state, xb, yb, lr, beta_one, beta_two, eps, step
        )
        losses.append(loss)
        step += 1
    return params, opt_state, step, losses

# Step 58 - train_loop
def init_opt_state(params):
    state = {}
    for layer in params:
        state[layer] = {}
        for pname in params[layer]:
            shape = params[layer][pname].shape
            state[layer][pname] = {
                "m": np.zeros(shape),
                "v": np.zeros(shape),
            }
    return state

def train_loop(params, x, y, num_epochs, batch_size,
                lr, beta_one, beta_two, eps, seed=0):
    opt_state = init_opt_state(params)
    step = 1
    loss_history = []
    for epoch in range(num_epochs):
        params, opt_state, step, losses = train_one_epoch(
            params, opt_state, x, y, batch_size,
            lr, beta_one, beta_two, eps, step, seed=seed + epoch
        )
        loss_history.extend(losses)
    return params, loss_history

# Step 59 - evaluate
def evaluate(params, x, y):
    logits, _ = lenet_forward(x, params)
    return accuracy(logits, y)

