import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import random
import numpy as np
class Model():
    def __init__(self,args,infer=False):
        self.args=args
        if infer:
            args.batch_size=1
            args.seq_length=1
        if args.model=='rnn':
            cell_fn=rnn_cell.BasicRNNCell
        elif args.model=='gru':
            cell_fn=rnn_cell.GRUCell
        elif args.model=='lstm':
            cell_fn=rnn_cell.BasicLSTMCell
        else:
            raise Exception("模型不支持：{}".format(args.model))

        cell=cell_fn(args.rnn_size)

        self.cell=cell=rnn_cell.MultiRNNCell([cell]*args.num_layers)

        self.input_data=tf.placeholder(tf.int32,[args.batch_size,args.seq_length]) #(10,25)

        self.targets=tf.placeholder(tf.int32,[args.batch_size,args.seq_length])

        self.initial_state=cell.zero_state(args.batch_size,tf.float32)

        
        #因为想要达到变量共享的效果, 就要在 tf.variable_scope()的作用域下使用 tf.get_variable() 这种方式产生和提取变量. 
        #不像 tf.Variable() 每次都会产生新的变量, tf.get_variable() 如果遇到了已经存在名字的变量时, 它会单纯的提取这个同样名字的变量，
        #如果不存在名字的变量再创建.
        with tf.variable_scope("rnnlm"):
            softmax_w=tf.get_variable("softmax_w",[args.rnn_size,args.vocab_size])  #args.vocab_size=19，19个方法
            softmax_b=tf.get_variable("softmax_b",[args.vocab_size])
            #attention=tf.get_variable("attention",[1,1,args.vocab_size])
            '''
            with tf.device("/cpu:0"):
                embedding=tf.get_variable("embedding",[args.vocab_size,args.rnn_size])
                
                #输入数据 self.input_data 的维度是 (batch_size , seq_length)
                #而输出的input_embedding 的维度成为 (batch_size ,num_steps ,rnn_size).   就是一个立方体,每个样例就是从头顶上削一片下来

                #词嵌入后成了这样一个三维数组，里面每一个元素是一个二维数组（25，32）
                temp=tf.nn.embedding_lookup(embedding,self.input_data)   #(10,25,32)

                
                #tf.split()函数将长方体按每一列切片，切成了25个片，每一片都是(10,32),表示这是这一批样本们的第t个特征，即在第xt时间步传入的input，embedding代替了ont-hot
                inputs=tf.split(1,args.seq_length,temp)   #len(inputs)=25
                #print(inputs[0].shape)    (10,1,32)
                #删除维度1  (10,32)   #每个数据从一列变成了一个扁平的长方形
                inputs=[tf.squeeze(input_,[1]) for input_ in inputs]
        '''
        '''
        def loop(prev,_):
            prev=tf.matmul(prev,softmax_w)+softmax_b
            
            #axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组
            prev_symbol=tf.stop_gradient(tf.argmax(prev,1))
            #stop_gradients也是一个list，list中的元素是tensorflow graph中的op，
            # 一旦进入这个list，将不会被计算梯度，更重要的是，在该op之后的BP计算都不会运行
            return tf.nn.embedding_lookup(embedding,prev_symbol)
        '''
        
        inputs=tf.split(1,args.seq_length,self.input_data)
        inputs=[tf.squeeze(input_,[1]) for input_ in inputs]

        #inputss=[tf.reshape(self.input_data[:,i],-1) for i in range(args.seq_length)]
        
        outputs,last_state=seq2seq.embedding_attention_seq2seq(inputs,inputs,cell,args.vocab_size,args.vocab_size,
                                                                args.rnn_size)

        #outputs,last_state=seq2seq.attention_decoder(inputs,self.initial_state, attention,cell,loop_function=loop if infer else None,scope='rnnlm')
        #outputs,last_state=seq2seq.rnn_decoder(inputs,self.initial_state,cell,loop_function=loop if infer else None,scope='rnnlm')
        
        self.saved_outputs=outputs 
        
        #print(len(outputs))  #是一个三维数组,有25个元素,对应步长，每个元素是一个二维数组(10，32)
        output=tf.reshape(tf.concat(1,outputs),[-1,args.vocab_size])
        #print(output)     //(250,32),将这25个(10,32)的二维数组按行堆叠了起来，行数变成了10*25
        
        #网络的最后输出(相当于最后添加了一个全连接层)
        #self.logits=tf.matmul(output,softmax_w)+softmax_b  #(250,19)
        self.logits=output
        #过一个softmax
        
        self.probs=tf.nn.softmax(self.logits)


        #参数要求：output [batch*numsteps, vocab_size]
        #target, [batch_size, num_steps]
        #weight:[tf.ones([batch_size * num_steps]
        #output具体的维度讲解见chrome"https://blog.csdn.net/xyz1584172808/article/details/83056179?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task"
        loss=seq2seq.sequence_loss_by_example([self.logits],[tf.reshape(self.targets,[-1])],[tf.ones([args.batch_size*args.seq_length])],args.vocab_size)


        self.cost=tf.reduce_sum(loss)/args.batch_size/args.seq_length
        self.final_state=last_state

        self.lr=tf.Variable(0.0,trainable=False)
        tvars=tf.trainable_variables()

        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,tvars),args.grad_clip)


        optimizer=tf.train.AdamOptimizer(self.lr)
        self.train_op=optimizer.apply_gradients(zip(grads,tvars))

        


