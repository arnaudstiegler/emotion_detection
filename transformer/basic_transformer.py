import torch
import torch.nn.functional as F 
from torch import nn

class Transformer(nn.Module):
	def __init__(self,k, heads, num_block, num_token, seq_length, num_classes):
		super().__init__()

		self.token_embedding = nn.Embedding(num_token, k)
		self.position_embedding = nn.Embedding(seq_length, k)

		transfo_block_list = []
		for i in range(num_block):
			transfo_block_list.append(TransformerBlock(k,heads))
		self.transfo_blocks = nn.Sequential(*transfo_block_list)

		self.to_probs = nn.Linear(k, num_classes)

	def forward(self, x):

		tokens = self.token_embedding(x)
		b, t, k = tokens.size()

		positions = torch.arange(t) #Compute the position vector

		#We have to make sure that the position embedding is the same across all elements of the batch
		positions = self.position_embedding(positions)[None, :, :].expand(b,t,k) 

		x = tokens + positions #We combine token and position embeddings into the same vector representation

		x = self.transfo_blocks(x)
		x = self.to_probs(x.mean(dim=1)) #We average the vectors of each sequence

		return F.softmax(x, dim = 1)



class TransformerBlock(nn.Module):
	def __init__(self, k ,heads=2):
		super().__init__()

		self.k, self.heads = k, heads

		self.self_attention = SelfAttention(self.k,self.heads)

		self.norm1 = nn.LayerNorm(k)
		self.norm2 = nn.LayerNorm(k)

		#We arbitrarily choose the number of hidden units for the first dense layer
		#The only constraint for this number is that it has to be bigger than the dimension (here, twice as big)
		self.ff = nn.Sequential(
					nn.Linear(k,2*k),
					nn.ReLU(),
					nn.Linear(2*k,k))

	def forward(self,x):

		attention_x = self.self_attention(x)
		x = self.norm1(x + attention_x)

		ff_x = self.ff(x)
		x = self.norm2(x + ff_x)

		return x


class SelfAttention(nn.Module):
	def __init__(self, k, heads=2):
		super().__init__()
		self.k, self.heads = k, heads

		self.to_keys = nn.Linear(k,k*heads) #We stack the Wk matrix for each head in the same matrix
		self.to_queries = nn.Linear(k,heads*k)
		self.to_values = nn.Linear(k, heads*k)

		self.merge_heads = nn.Linear(heads*k,k) #After the heads, we merge the result

	def forward(self,x):
		b,t,k = x.size()
		h = self.heads

		#We basically give each head its own dimension (third one)
		keys = self.to_keys(x).view(b,t,h,k) #batch,seq_length,nb_heads,emb_dimension
		queries = self.to_queries(x).view(b,t,h,k)
		values = self.to_values(x).view(b,t,h,k)

		keys = keys.transpose(1,2).contiguous().view(b*h,t,k) #transpose used to collapse heads into batch
		queries = queries.transpose(1,2).contiguous().view(b*h,t,k)
		values = values.transpose(1,2).contiguous().view(b*h,t,k)

		queries = queries // (k)**(1/4) #We rescale now so that we don't have to do it after the dot product (faster)
		keys = keys // (k)**(1/4)

		dot = torch.bmm(queries,keys.transpose(1,2))
		dot = F.softmax(dot,dim=2)

		out = torch.bmm(dot,values)

		out = out.contiguous().view(b,h,t,k)
		out = out.transpose(1,2).reshape(b,t,h*k)

		return self.merge_heads(out)



