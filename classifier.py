model=MaterialClassifier(lin_output_size, lin_input_size)

for comb in range(0,1):
    total_aux_labels=dict(total_labels)#aux variable to take elements out for random implementation
    for k in range(0,4):
            index_taken_input=givemeIndex(total_aux_labels,comb+k)
            trasf_matrix=PCA(total_folds,total_labels,index_taken_input)
            log_probs = model(autograd.Variable(trasf_matrix.t()))
            optimizer = optim.SGD(model.parameters(), lr=0.5)
            learning_rate=20
            input_v=torch.Tensor(trasf_matrix.shape[1],1,trasf_matrix.shape[0])
            input_v[:,0,:]=trasf_matrix.t()
            input_variable=autograd.Variable(input_v,requires_grad=True)
            target=autograd.Variable(total_labels[index_taken_input],requires_grad=False)

    for z in (0,1):
    	out=autograd.Variable(torch.Tensor(1,10))
    	out=model(autograd.Variable(input_v,requires_grad=True))
    	predict_tensors=torch.FloatTensor(41475,1).zero_()
    	for j in range(len(out)):
        	index_numpy=total_labels[k][j].numpy()
        	predict_tensors[j]=out[j,index_numpy[0]].data
        	predict_tensors[j]=out[j,label_vector_tensor[k][j].data]
   		out=out.t()
            	targetino=autograd.Variable(total_labels[k].long())
            	targetin=autograd.Variable(torch.ones(1).long())
        
            for y in range(0,len(input_v)-1):
        	print("input:",input_variable[y])
                model.zero_grad()
                while(1):
            	    model.zero_grad()
                    uscita=model(input_variable[y])
                    output =F.nll_loss(uscita,target[y][0])
                    output.backward()
                    for p in model.parameters():
                        p.data.sub_(p.grad.data * learning_rate)
                    	if output.data[0]==0: break
       
    index_taken_input=givemeIndex(total_aux_labels,comb+k+1)
    trasf_matrix=PCA(total_folds,total_labels,index_taken_input)
    input_v=torch.Tensor(trasf_matrix.shape[1],1,trasf_matrix.shape[0])
    input_v[:,0,:]=trasf_matrix.t()
    input_variable=autograd.Variable(input_v,requires_grad=True)
    input_v = autograd.Variable(trasf_matrix.t(),requires_grad=True)
    target=autograd.Variable(total_labels[index_taken_input],requires_grad=False)
    comparing_tensor=torch.Tensor(41475,11)
    for y in range(0,len(input_v)-1):

                uscita=model(input_variable[y])
                openfile.write(str(uscita.data))
                openfile.write(str(target[y][0]))
                output =F.nll_loss(uscita,target[y][0])
                comparing_tensor[y,0:10]=uscita.data
                comparing_tensor[y,10]=target[y][0].data[0]
