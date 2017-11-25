import lasagne
import cascadenet.network.layers as l
from collections import OrderedDict



def cascade_resnet(pr, net, input_layer, n=5, nf=64, b=lasagne.init.Constant, **kwargs):
    shape = lasagne.layers.get_output_shape(input_layer)
    n_channel = shape[1]
    net[pr+'conv1'] = l.Conv(input_layer, nf, 3,
                             b=b(),
                             name=pr+'conv1')

    for i in range(2, n):
        net[pr+'conv%d'%i] = l.Conv(net[pr+'conv%d'%(i-1)], nf, 3, b=b(),
                                    name=pr+'conv%d'%i)

    net[pr+'conv_aggr'] = l.ConvAggr(net[pr+'conv%d'%(n-1)], n_channel, 3,
                                     b=b(), name=pr+'conv_aggr')
    net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer],
                                    name=pr+'res')
    output_layer = net[pr+'res']
    return net, output_layer

#work version
def cascade_Unet(pr, net, input_layer, n=5, nf=64, b=lasagne.init.Constant, **kwargs):
    shape = lasagne.layers.get_output_shape(input_layer)
    n_channel = shape[1]
    net[pr+'conv1'] = l.Conv(input_layer, nf, 3,
                             b=b(),
                             name=pr+'conv1')
	
	#changed this part from cascade_resnet
	# solved: (contracting path -> sizes need to be equal), (padding problems -> need to have same size in the end -> stride and padding parameters do not change problem)
    net[pr+'conv2'] = l.Conv(net[pr+'conv1'], nf, 3, b=b(),
                                    name=pr+'conv2')
    net[pr+'max1'] = l.MaxPool(net[pr+'conv2'], 2, stride=2, pad=(0,0), name=pr+'max1')
	
	
    net[pr+'conv3'] = l.Conv(net[pr+'max1'], nf*2, 3, b=b(),
                                    name=pr+'conv3')
    net[pr+'conv4'] = l.Conv(net[pr+'conv3'], nf*2, 3, b=b(),
                                    name=pr+'conv4')
    net[pr+'max2'] = l.MaxPool(net[pr+'conv4'], 2, stride=2, pad=(1,1), name=pr+'max2')
	
	
    net[pr+'conv5'] = l.Conv(net[pr+'max2'], nf*4, 3, b=b(),
                                    name=pr+'conv5')
    net[pr+'conv6'] = l.Conv(net[pr+'conv5'], nf*4, 3, b=b(),
                                    name=pr+'conv6')
    net[pr+'deconv1'] = l.Deconv(net[pr+'conv6'], nf*2, 2, stride=2, crop='full', b=b(), name=pr+'deconv1')
    
    #shape1 = lasagne.layers.get_output_shape(net[pr+'conv3'])
    #shape2 = lasagne.layers.get_output_shape(net[pr+'deconv1'])
    #print(shape1)
    #print(shape2)
	
	#additional input conv4 (contracting path)
    net[pr+'concat1']=lasagne.layers.concat([net[pr+'deconv1'], net[pr+'conv4']])
    net[pr+'conv7'] = l.Conv(net[pr+'concat1'], nf*2, 3, b=b(),
                                    name=pr+'conv7')
    net[pr+'conv8'] = l.Conv(net[pr+'conv7'], nf*2, 3, b=b(),
                                    name=pr+'conv8')
    net[pr+'deconv2'] = l.Deconv(net[pr+'conv8'], nf, 2, stride=2, crop='valid', b=b(), name=pr+'deconv2')
    
    #shape3 = lasagne.layers.get_output_shape(net[pr+'conv2'])
    #shape4 = lasagne.layers.get_output_shape(net[pr+'deconv2'])
    #print(shape3)
    #print(shape4)
	
	#additional input conv2 (contracting path)
    net[pr+'concat2']=lasagne.layers.concat([net[pr+'deconv2'], net[pr+'conv2']])
    net[pr+'conv9'] = l.Conv(net[pr+'concat2'], nf, 3, b=b(),
                                    name=pr+'conv9')
    net[pr+'conv10'] = l.Conv(net[pr+'conv9'], nf, 3, b=b(),
                                    name=pr+'conv10')

    net[pr+'conv_aggr'] = l.ConvAggr(net[pr+'conv10'], n_channel, 3, pad='same',
                                     b=b(), name=pr+'conv_aggr')
    net[pr+'res'] = l.ResidualLayer([net[pr+'conv_aggr'], input_layer],
                                    name=pr+'res')
    output_layer = lasagne.layers.reshape(net[pr+'res'],shape)
    
    return net, output_layer

def build_cascade_cnn_from_list(shape, net_meta):
    """
    Create iterative network with more flexibility

    net_meta: [(model1, cascade1_n),(model2, cascade2_n),....(modelm, cascadem_n),]
    """
    if not net_meta:
        raise

    net = OrderedDict()
    input_layer, kspace_input_layer, mask_layer = l.get_dc_input_layers(shape)
    net['input'] = input_layer
    net['kspace_input'] = kspace_input_layer
    net['mask'] = mask_layer

    j = 0
    for cascade_net, cascade_n in net_meta:
        # Cascade layer
        print('hi')
        for i in range(cascade_n):
            pr = 'c%d_' % j
            print(pr, i)
            net, output_layer = cascade_net(pr, net, input_layer)
            #print(shape)
            
			
            # add data consistency layer
			# (make sure it has the same size as the output of the previous U-net)
            net[pr+'dc'] = l.DCLayer([output_layer,
                                      net['mask'],
                                      net['kspace_input']],
                                     shape)
            input_layer = net[pr+'dc']
            j += 1

    output_layer = input_layer

    return net, output_layer


def build_d2_c2(shape):
    def cascade_d2(pr, net, input_layer, **kwargs):
        return cascade_resnet(pr, net, input_layer, n=2)
    return build_cascade_cnn_from_list(shape, [(cascade_d2, 2)])
	
def build_UnetCascade(shape):
    def cascade_d2(pr, net, input_layer, **kwargs):
        return cascade_Unet(pr, net, input_layer, n=18)
    return build_cascade_cnn_from_list(shape, [(cascade_d2, 3)]) 

def build_d5_c5(shape):
    return build_cascade_cnn_from_list(shape, [(cascade_resnet, 5)])
