��
�&�&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
K
Bincount
arr
size
weights"T	
bins"T"
Ttype:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
Cumsum
x"T
axis"Tidx
out"T"
	exclusivebool( "
reversebool( " 
Ttype:
2	"
Tidxtype0:
2	
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
�
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype�
.
Identity

input"T
output"T"	
Ttype
�
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0���������"
value_indexint(0���������"+

vocab_sizeint���������(0���������"
	delimiterstring	"
offsetint �
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype�
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RaggedTensorToTensor
shape"Tshape
values"T
default_value"T:
row_partition_tensors"Tindex*num_row_partition_tensors
result"T"	
Ttype"
Tindextype:
2	"
Tshapetype:
2	"$
num_row_partition_tensorsint(0"#
row_partition_typeslist(string)
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
e
StringSplitV2	
input
sep
indices	

values	
shape	"
maxsplitint���������
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.7.12unknown8��
�
#deep_cocrystal/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:! *4
shared_name%#deep_cocrystal/embedding/embeddings
�
7deep_cocrystal/embedding/embeddings/Read/ReadVariableOpReadVariableOp#deep_cocrystal/embedding/embeddings*
_output_shapes

:! *
dtype0
�
%deep_cocrystal/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:! *6
shared_name'%deep_cocrystal/embedding_1/embeddings
�
9deep_cocrystal/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp%deep_cocrystal/embedding_1/embeddings*
_output_shapes

:! *
dtype0
�
%deep_cocrystal/prediction_head/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*6
shared_name'%deep_cocrystal/prediction_head/kernel
�
9deep_cocrystal/prediction_head/kernel/Read/ReadVariableOpReadVariableOp%deep_cocrystal/prediction_head/kernel*
_output_shapes
:	�*
dtype0
�
#deep_cocrystal/prediction_head/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#deep_cocrystal/prediction_head/bias
�
7deep_cocrystal/prediction_head/bias/Read/ReadVariableOpReadVariableOp#deep_cocrystal/prediction_head/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
deep_cocrystal/api_cnn_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*0
shared_name!deep_cocrystal/api_cnn_0/kernel
�
3deep_cocrystal/api_cnn_0/kernel/Read/ReadVariableOpReadVariableOpdeep_cocrystal/api_cnn_0/kernel*#
_output_shapes
: �*
dtype0
�
deep_cocrystal/api_cnn_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namedeep_cocrystal/api_cnn_0/bias
�
1deep_cocrystal/api_cnn_0/bias/Read/ReadVariableOpReadVariableOpdeep_cocrystal/api_cnn_0/bias*
_output_shapes	
:�*
dtype0
�
deep_cocrystal/api_cnn_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!deep_cocrystal/api_cnn_1/kernel
�
3deep_cocrystal/api_cnn_1/kernel/Read/ReadVariableOpReadVariableOpdeep_cocrystal/api_cnn_1/kernel*$
_output_shapes
:��*
dtype0
�
deep_cocrystal/api_cnn_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namedeep_cocrystal/api_cnn_1/bias
�
1deep_cocrystal/api_cnn_1/bias/Read/ReadVariableOpReadVariableOpdeep_cocrystal/api_cnn_1/bias*
_output_shapes	
:�*
dtype0
�
deep_cocrystal/cof_cnn_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*0
shared_name!deep_cocrystal/cof_cnn_0/kernel
�
3deep_cocrystal/cof_cnn_0/kernel/Read/ReadVariableOpReadVariableOpdeep_cocrystal/cof_cnn_0/kernel*#
_output_shapes
: �*
dtype0
�
deep_cocrystal/cof_cnn_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namedeep_cocrystal/cof_cnn_0/bias
�
1deep_cocrystal/cof_cnn_0/bias/Read/ReadVariableOpReadVariableOpdeep_cocrystal/cof_cnn_0/bias*
_output_shapes	
:�*
dtype0
�
deep_cocrystal/cof_cnn_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*0
shared_name!deep_cocrystal/cof_cnn_1/kernel
�
3deep_cocrystal/cof_cnn_1/kernel/Read/ReadVariableOpReadVariableOpdeep_cocrystal/cof_cnn_1/kernel*$
_output_shapes
:��*
dtype0
�
deep_cocrystal/cof_cnn_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_namedeep_cocrystal/cof_cnn_1/bias
�
1deep_cocrystal/cof_cnn_1/bias/Read/ReadVariableOpReadVariableOpdeep_cocrystal/cof_cnn_1/bias*
_output_shapes	
:�*
dtype0
�
)deep_cocrystal/interaction_dense_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*:
shared_name+)deep_cocrystal/interaction_dense_0/kernel
�
=deep_cocrystal/interaction_dense_0/kernel/Read/ReadVariableOpReadVariableOp)deep_cocrystal/interaction_dense_0/kernel* 
_output_shapes
:
��*
dtype0
�
'deep_cocrystal/interaction_dense_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'deep_cocrystal/interaction_dense_0/bias
�
;deep_cocrystal/interaction_dense_0/bias/Read/ReadVariableOpReadVariableOp'deep_cocrystal/interaction_dense_0/bias*
_output_shapes	
:�*
dtype0
�
)deep_cocrystal/interaction_dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*:
shared_name+)deep_cocrystal/interaction_dense_1/kernel
�
=deep_cocrystal/interaction_dense_1/kernel/Read/ReadVariableOpReadVariableOp)deep_cocrystal/interaction_dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
'deep_cocrystal/interaction_dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'deep_cocrystal/interaction_dense_1/bias
�
;deep_cocrystal/interaction_dense_1/bias/Read/ReadVariableOpReadVariableOp'deep_cocrystal/interaction_dense_1/bias*
_output_shapes	
:�*
dtype0
�
)deep_cocrystal/interaction_dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*:
shared_name+)deep_cocrystal/interaction_dense_2/kernel
�
=deep_cocrystal/interaction_dense_2/kernel/Read/ReadVariableOpReadVariableOp)deep_cocrystal/interaction_dense_2/kernel* 
_output_shapes
:
��*
dtype0
�
'deep_cocrystal/interaction_dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*8
shared_name)'deep_cocrystal/interaction_dense_2/bias
�
;deep_cocrystal/interaction_dense_2/bias/Read/ReadVariableOpReadVariableOp'deep_cocrystal/interaction_dense_2/bias*
_output_shapes	
:�*
dtype0
�

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*9
shared_name*(hash_table_./data/vocabulary.txt_-2_-1_2*
value_dtype0	
�
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*9
shared_name*(hash_table_./data/vocabulary.txt_-2_-1_2*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
R
Variable/AssignAssignVariableOpVariableasset_path_initializer*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
X
Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
�
*Adam/deep_cocrystal/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:! *;
shared_name,*Adam/deep_cocrystal/embedding/embeddings/m
�
>Adam/deep_cocrystal/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp*Adam/deep_cocrystal/embedding/embeddings/m*
_output_shapes

:! *
dtype0
�
,Adam/deep_cocrystal/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:! *=
shared_name.,Adam/deep_cocrystal/embedding_1/embeddings/m
�
@Adam/deep_cocrystal/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp,Adam/deep_cocrystal/embedding_1/embeddings/m*
_output_shapes

:! *
dtype0
�
,Adam/deep_cocrystal/prediction_head/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*=
shared_name.,Adam/deep_cocrystal/prediction_head/kernel/m
�
@Adam/deep_cocrystal/prediction_head/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/deep_cocrystal/prediction_head/kernel/m*
_output_shapes
:	�*
dtype0
�
*Adam/deep_cocrystal/prediction_head/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/deep_cocrystal/prediction_head/bias/m
�
>Adam/deep_cocrystal/prediction_head/bias/m/Read/ReadVariableOpReadVariableOp*Adam/deep_cocrystal/prediction_head/bias/m*
_output_shapes
:*
dtype0
�
&Adam/deep_cocrystal/api_cnn_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*7
shared_name(&Adam/deep_cocrystal/api_cnn_0/kernel/m
�
:Adam/deep_cocrystal/api_cnn_0/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/api_cnn_0/kernel/m*#
_output_shapes
: �*
dtype0
�
$Adam/deep_cocrystal/api_cnn_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/api_cnn_0/bias/m
�
8Adam/deep_cocrystal/api_cnn_0/bias/m/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/api_cnn_0/bias/m*
_output_shapes	
:�*
dtype0
�
&Adam/deep_cocrystal/api_cnn_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*7
shared_name(&Adam/deep_cocrystal/api_cnn_1/kernel/m
�
:Adam/deep_cocrystal/api_cnn_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/api_cnn_1/kernel/m*$
_output_shapes
:��*
dtype0
�
$Adam/deep_cocrystal/api_cnn_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/api_cnn_1/bias/m
�
8Adam/deep_cocrystal/api_cnn_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/api_cnn_1/bias/m*
_output_shapes	
:�*
dtype0
�
&Adam/deep_cocrystal/cof_cnn_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*7
shared_name(&Adam/deep_cocrystal/cof_cnn_0/kernel/m
�
:Adam/deep_cocrystal/cof_cnn_0/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/cof_cnn_0/kernel/m*#
_output_shapes
: �*
dtype0
�
$Adam/deep_cocrystal/cof_cnn_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/cof_cnn_0/bias/m
�
8Adam/deep_cocrystal/cof_cnn_0/bias/m/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/cof_cnn_0/bias/m*
_output_shapes	
:�*
dtype0
�
&Adam/deep_cocrystal/cof_cnn_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*7
shared_name(&Adam/deep_cocrystal/cof_cnn_1/kernel/m
�
:Adam/deep_cocrystal/cof_cnn_1/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/cof_cnn_1/kernel/m*$
_output_shapes
:��*
dtype0
�
$Adam/deep_cocrystal/cof_cnn_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/cof_cnn_1/bias/m
�
8Adam/deep_cocrystal/cof_cnn_1/bias/m/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/cof_cnn_1/bias/m*
_output_shapes	
:�*
dtype0
�
0Adam/deep_cocrystal/interaction_dense_0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*A
shared_name20Adam/deep_cocrystal/interaction_dense_0/kernel/m
�
DAdam/deep_cocrystal/interaction_dense_0/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/deep_cocrystal/interaction_dense_0/kernel/m* 
_output_shapes
:
��*
dtype0
�
.Adam/deep_cocrystal/interaction_dense_0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.Adam/deep_cocrystal/interaction_dense_0/bias/m
�
BAdam/deep_cocrystal/interaction_dense_0/bias/m/Read/ReadVariableOpReadVariableOp.Adam/deep_cocrystal/interaction_dense_0/bias/m*
_output_shapes	
:�*
dtype0
�
0Adam/deep_cocrystal/interaction_dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*A
shared_name20Adam/deep_cocrystal/interaction_dense_1/kernel/m
�
DAdam/deep_cocrystal/interaction_dense_1/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/deep_cocrystal/interaction_dense_1/kernel/m* 
_output_shapes
:
��*
dtype0
�
.Adam/deep_cocrystal/interaction_dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.Adam/deep_cocrystal/interaction_dense_1/bias/m
�
BAdam/deep_cocrystal/interaction_dense_1/bias/m/Read/ReadVariableOpReadVariableOp.Adam/deep_cocrystal/interaction_dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
0Adam/deep_cocrystal/interaction_dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*A
shared_name20Adam/deep_cocrystal/interaction_dense_2/kernel/m
�
DAdam/deep_cocrystal/interaction_dense_2/kernel/m/Read/ReadVariableOpReadVariableOp0Adam/deep_cocrystal/interaction_dense_2/kernel/m* 
_output_shapes
:
��*
dtype0
�
.Adam/deep_cocrystal/interaction_dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.Adam/deep_cocrystal/interaction_dense_2/bias/m
�
BAdam/deep_cocrystal/interaction_dense_2/bias/m/Read/ReadVariableOpReadVariableOp.Adam/deep_cocrystal/interaction_dense_2/bias/m*
_output_shapes	
:�*
dtype0
�
*Adam/deep_cocrystal/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:! *;
shared_name,*Adam/deep_cocrystal/embedding/embeddings/v
�
>Adam/deep_cocrystal/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp*Adam/deep_cocrystal/embedding/embeddings/v*
_output_shapes

:! *
dtype0
�
,Adam/deep_cocrystal/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:! *=
shared_name.,Adam/deep_cocrystal/embedding_1/embeddings/v
�
@Adam/deep_cocrystal/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp,Adam/deep_cocrystal/embedding_1/embeddings/v*
_output_shapes

:! *
dtype0
�
,Adam/deep_cocrystal/prediction_head/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*=
shared_name.,Adam/deep_cocrystal/prediction_head/kernel/v
�
@Adam/deep_cocrystal/prediction_head/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/deep_cocrystal/prediction_head/kernel/v*
_output_shapes
:	�*
dtype0
�
*Adam/deep_cocrystal/prediction_head/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/deep_cocrystal/prediction_head/bias/v
�
>Adam/deep_cocrystal/prediction_head/bias/v/Read/ReadVariableOpReadVariableOp*Adam/deep_cocrystal/prediction_head/bias/v*
_output_shapes
:*
dtype0
�
&Adam/deep_cocrystal/api_cnn_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*7
shared_name(&Adam/deep_cocrystal/api_cnn_0/kernel/v
�
:Adam/deep_cocrystal/api_cnn_0/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/api_cnn_0/kernel/v*#
_output_shapes
: �*
dtype0
�
$Adam/deep_cocrystal/api_cnn_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/api_cnn_0/bias/v
�
8Adam/deep_cocrystal/api_cnn_0/bias/v/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/api_cnn_0/bias/v*
_output_shapes	
:�*
dtype0
�
&Adam/deep_cocrystal/api_cnn_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*7
shared_name(&Adam/deep_cocrystal/api_cnn_1/kernel/v
�
:Adam/deep_cocrystal/api_cnn_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/api_cnn_1/kernel/v*$
_output_shapes
:��*
dtype0
�
$Adam/deep_cocrystal/api_cnn_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/api_cnn_1/bias/v
�
8Adam/deep_cocrystal/api_cnn_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/api_cnn_1/bias/v*
_output_shapes	
:�*
dtype0
�
&Adam/deep_cocrystal/cof_cnn_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: �*7
shared_name(&Adam/deep_cocrystal/cof_cnn_0/kernel/v
�
:Adam/deep_cocrystal/cof_cnn_0/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/cof_cnn_0/kernel/v*#
_output_shapes
: �*
dtype0
�
$Adam/deep_cocrystal/cof_cnn_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/cof_cnn_0/bias/v
�
8Adam/deep_cocrystal/cof_cnn_0/bias/v/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/cof_cnn_0/bias/v*
_output_shapes	
:�*
dtype0
�
&Adam/deep_cocrystal/cof_cnn_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*7
shared_name(&Adam/deep_cocrystal/cof_cnn_1/kernel/v
�
:Adam/deep_cocrystal/cof_cnn_1/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/deep_cocrystal/cof_cnn_1/kernel/v*$
_output_shapes
:��*
dtype0
�
$Adam/deep_cocrystal/cof_cnn_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$Adam/deep_cocrystal/cof_cnn_1/bias/v
�
8Adam/deep_cocrystal/cof_cnn_1/bias/v/Read/ReadVariableOpReadVariableOp$Adam/deep_cocrystal/cof_cnn_1/bias/v*
_output_shapes	
:�*
dtype0
�
0Adam/deep_cocrystal/interaction_dense_0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*A
shared_name20Adam/deep_cocrystal/interaction_dense_0/kernel/v
�
DAdam/deep_cocrystal/interaction_dense_0/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/deep_cocrystal/interaction_dense_0/kernel/v* 
_output_shapes
:
��*
dtype0
�
.Adam/deep_cocrystal/interaction_dense_0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.Adam/deep_cocrystal/interaction_dense_0/bias/v
�
BAdam/deep_cocrystal/interaction_dense_0/bias/v/Read/ReadVariableOpReadVariableOp.Adam/deep_cocrystal/interaction_dense_0/bias/v*
_output_shapes	
:�*
dtype0
�
0Adam/deep_cocrystal/interaction_dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*A
shared_name20Adam/deep_cocrystal/interaction_dense_1/kernel/v
�
DAdam/deep_cocrystal/interaction_dense_1/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/deep_cocrystal/interaction_dense_1/kernel/v* 
_output_shapes
:
��*
dtype0
�
.Adam/deep_cocrystal/interaction_dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.Adam/deep_cocrystal/interaction_dense_1/bias/v
�
BAdam/deep_cocrystal/interaction_dense_1/bias/v/Read/ReadVariableOpReadVariableOp.Adam/deep_cocrystal/interaction_dense_1/bias/v*
_output_shapes	
:�*
dtype0
�
0Adam/deep_cocrystal/interaction_dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*A
shared_name20Adam/deep_cocrystal/interaction_dense_2/kernel/v
�
DAdam/deep_cocrystal/interaction_dense_2/kernel/v/Read/ReadVariableOpReadVariableOp0Adam/deep_cocrystal/interaction_dense_2/kernel/v* 
_output_shapes
:
��*
dtype0
�
.Adam/deep_cocrystal/interaction_dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*?
shared_name0.Adam/deep_cocrystal/interaction_dense_2/bias/v
�
BAdam/deep_cocrystal/interaction_dense_2/bias/v/Read/ReadVariableOpReadVariableOp.Adam/deep_cocrystal/interaction_dense_2/bias/v*
_output_shapes	
:�*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R
H
Const_4Const*
_output_shapes
: *
dtype0*
valueB B 
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R 
a
ReadVariableOpReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
�
StatefulPartitionedCallStatefulPartitionedCallReadVariableOp
hash_table*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *"
fR
__inference_<lambda>_9074
g
ReadVariableOp_1ReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
�
StatefulPartitionedCall_1StatefulPartitionedCallReadVariableOp_1hash_table_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *"
fR
__inference_<lambda>_9081
h
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1^Variable/Assign^Variable_1/Assign
�p
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*�o
value�oB�o B�o
�
api_smiles_encoder
cof_smiles_encoder
api_embedding
coformer_embedding
	api_convs
	cof_convs
api_pooling
cof_pooling
	interaction_concat

interaction_denses
prediction_head
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
"
_lookup_layer
	keras_api
"
_lookup_layer
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api

 0
!1

"0
#1
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*
00
11
22
33
44
55
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
�
<iter

=beta_1

>beta_2
	?decay
@learning_ratem�m�6m�7m�Am�Bm�Cm�Dm�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�Mm�Nm�v�v�6v�7v�Av�Bv�Cv�Dv�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�Mv�Nv�
�
0
1
A2
B3
C4
D5
E6
F7
G8
H9
I10
J11
K12
L13
M14
N15
616
717
�
0
1
A2
B3
C4
D5
E6
F7
G8
H9
I10
J11
K12
L13
M14
N15
616
717
 
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
 
!
Tlookup_table
U	keras_api
 
!
Vlookup_table
W	keras_api
 
lj
VARIABLE_VALUE#deep_cocrystal/embedding/embeddings3api_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
sq
VARIABLE_VALUE%deep_cocrystal/embedding_1/embeddings8coformer_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
h

Akernel
Bbias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
h

Ckernel
Dbias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

Ekernel
Fbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
h

Gkernel
Hbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
 
 
 
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
$	variables
%trainable_variables
&regularization_losses
 
 
 
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
(	variables
)trainable_variables
*regularization_losses
 
 
 
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
l

Ikernel
Jbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
l

Kkernel
Lbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
l

Mkernel
Nbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
V
�	variables
�trainable_variables
�regularization_losses
�	keras_api
lj
VARIABLE_VALUE%deep_cocrystal/prediction_head/kernel1prediction_head/kernel/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE#deep_cocrystal/prediction_head/bias/prediction_head/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdeep_cocrystal/api_cnn_0/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeep_cocrystal/api_cnn_0/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdeep_cocrystal/api_cnn_1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeep_cocrystal/api_cnn_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdeep_cocrystal/cof_cnn_0/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeep_cocrystal/cof_cnn_0/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEdeep_cocrystal/cof_cnn_1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdeep_cocrystal/cof_cnn_1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)deep_cocrystal/interaction_dense_0/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'deep_cocrystal/interaction_dense_0/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)deep_cocrystal/interaction_dense_1/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'deep_cocrystal/interaction_dense_1/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)deep_cocrystal/interaction_dense_2/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'deep_cocrystal/interaction_dense_2/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
 
�
0
1
2
3
 4
!5
"6
#7
8
9
	10
011
112
213
314
415
516
17

�0
�1
 
 

�_initializer
 

�_initializer
 
 
 
 
 
 
 
 
 
 
 

A0
B1

A0
B1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses

C0
D1

C0
D1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses

E0
F1

E0
F1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses

G0
H1

G0
H1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

I0
J1

I0
J1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

K0
L1

K0
L1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses

M0
N1

M0
N1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api

�	_filename

�	_filename
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
 
 
��
VARIABLE_VALUE*Adam/deep_cocrystal/embedding/embeddings/mOapi_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/deep_cocrystal/embedding_1/embeddings/mTcoformer_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/deep_cocrystal/prediction_head/kernel/mMprediction_head/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/deep_cocrystal/prediction_head/bias/mKprediction_head/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/api_cnn_0/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/api_cnn_0/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/api_cnn_1/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/api_cnn_1/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/cof_cnn_0/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/cof_cnn_0/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/cof_cnn_1/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/cof_cnn_1/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/deep_cocrystal/interaction_dense_0/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.Adam/deep_cocrystal/interaction_dense_0/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/deep_cocrystal/interaction_dense_1/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.Adam/deep_cocrystal/interaction_dense_1/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/deep_cocrystal/interaction_dense_2/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.Adam/deep_cocrystal/interaction_dense_2/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/deep_cocrystal/embedding/embeddings/vOapi_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/deep_cocrystal/embedding_1/embeddings/vTcoformer_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE,Adam/deep_cocrystal/prediction_head/kernel/vMprediction_head/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*Adam/deep_cocrystal/prediction_head/bias/vKprediction_head/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/api_cnn_0/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/api_cnn_0/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/api_cnn_1/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/api_cnn_1/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/cof_cnn_0/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/cof_cnn_0/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUE&Adam/deep_cocrystal/cof_cnn_1/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE$Adam/deep_cocrystal/cof_cnn_1/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/deep_cocrystal/interaction_dense_0/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.Adam/deep_cocrystal/interaction_dense_0/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/deep_cocrystal/interaction_dense_1/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.Adam/deep_cocrystal/interaction_dense_1/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0Adam/deep_cocrystal/interaction_dense_2/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.Adam/deep_cocrystal/interaction_dense_2/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
r
serving_default_input_1Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
r
serving_default_input_2Placeholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCall_2StatefulPartitionedCallserving_default_input_1serving_default_input_2
hash_tableConstConst_1Const_2hash_table_1Const_3Const_4Const_5#deep_cocrystal/embedding/embeddings%deep_cocrystal/embedding_1/embeddingsdeep_cocrystal/api_cnn_0/kerneldeep_cocrystal/api_cnn_0/biasdeep_cocrystal/cof_cnn_0/kerneldeep_cocrystal/cof_cnn_0/biasdeep_cocrystal/api_cnn_1/kerneldeep_cocrystal/api_cnn_1/biasdeep_cocrystal/cof_cnn_1/kerneldeep_cocrystal/cof_cnn_1/bias)deep_cocrystal/interaction_dense_0/kernel'deep_cocrystal/interaction_dense_0/bias)deep_cocrystal/interaction_dense_1/kernel'deep_cocrystal/interaction_dense_1/bias)deep_cocrystal/interaction_dense_2/kernel'deep_cocrystal/interaction_dense_2/bias%deep_cocrystal/prediction_head/kernel#deep_cocrystal/prediction_head/bias*'
Tin 
2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference_signature_wrapper_8158
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename7deep_cocrystal/embedding/embeddings/Read/ReadVariableOp9deep_cocrystal/embedding_1/embeddings/Read/ReadVariableOp9deep_cocrystal/prediction_head/kernel/Read/ReadVariableOp7deep_cocrystal/prediction_head/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp3deep_cocrystal/api_cnn_0/kernel/Read/ReadVariableOp1deep_cocrystal/api_cnn_0/bias/Read/ReadVariableOp3deep_cocrystal/api_cnn_1/kernel/Read/ReadVariableOp1deep_cocrystal/api_cnn_1/bias/Read/ReadVariableOp3deep_cocrystal/cof_cnn_0/kernel/Read/ReadVariableOp1deep_cocrystal/cof_cnn_0/bias/Read/ReadVariableOp3deep_cocrystal/cof_cnn_1/kernel/Read/ReadVariableOp1deep_cocrystal/cof_cnn_1/bias/Read/ReadVariableOp=deep_cocrystal/interaction_dense_0/kernel/Read/ReadVariableOp;deep_cocrystal/interaction_dense_0/bias/Read/ReadVariableOp=deep_cocrystal/interaction_dense_1/kernel/Read/ReadVariableOp;deep_cocrystal/interaction_dense_1/bias/Read/ReadVariableOp=deep_cocrystal/interaction_dense_2/kernel/Read/ReadVariableOp;deep_cocrystal/interaction_dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp>Adam/deep_cocrystal/embedding/embeddings/m/Read/ReadVariableOp@Adam/deep_cocrystal/embedding_1/embeddings/m/Read/ReadVariableOp@Adam/deep_cocrystal/prediction_head/kernel/m/Read/ReadVariableOp>Adam/deep_cocrystal/prediction_head/bias/m/Read/ReadVariableOp:Adam/deep_cocrystal/api_cnn_0/kernel/m/Read/ReadVariableOp8Adam/deep_cocrystal/api_cnn_0/bias/m/Read/ReadVariableOp:Adam/deep_cocrystal/api_cnn_1/kernel/m/Read/ReadVariableOp8Adam/deep_cocrystal/api_cnn_1/bias/m/Read/ReadVariableOp:Adam/deep_cocrystal/cof_cnn_0/kernel/m/Read/ReadVariableOp8Adam/deep_cocrystal/cof_cnn_0/bias/m/Read/ReadVariableOp:Adam/deep_cocrystal/cof_cnn_1/kernel/m/Read/ReadVariableOp8Adam/deep_cocrystal/cof_cnn_1/bias/m/Read/ReadVariableOpDAdam/deep_cocrystal/interaction_dense_0/kernel/m/Read/ReadVariableOpBAdam/deep_cocrystal/interaction_dense_0/bias/m/Read/ReadVariableOpDAdam/deep_cocrystal/interaction_dense_1/kernel/m/Read/ReadVariableOpBAdam/deep_cocrystal/interaction_dense_1/bias/m/Read/ReadVariableOpDAdam/deep_cocrystal/interaction_dense_2/kernel/m/Read/ReadVariableOpBAdam/deep_cocrystal/interaction_dense_2/bias/m/Read/ReadVariableOp>Adam/deep_cocrystal/embedding/embeddings/v/Read/ReadVariableOp@Adam/deep_cocrystal/embedding_1/embeddings/v/Read/ReadVariableOp@Adam/deep_cocrystal/prediction_head/kernel/v/Read/ReadVariableOp>Adam/deep_cocrystal/prediction_head/bias/v/Read/ReadVariableOp:Adam/deep_cocrystal/api_cnn_0/kernel/v/Read/ReadVariableOp8Adam/deep_cocrystal/api_cnn_0/bias/v/Read/ReadVariableOp:Adam/deep_cocrystal/api_cnn_1/kernel/v/Read/ReadVariableOp8Adam/deep_cocrystal/api_cnn_1/bias/v/Read/ReadVariableOp:Adam/deep_cocrystal/cof_cnn_0/kernel/v/Read/ReadVariableOp8Adam/deep_cocrystal/cof_cnn_0/bias/v/Read/ReadVariableOp:Adam/deep_cocrystal/cof_cnn_1/kernel/v/Read/ReadVariableOp8Adam/deep_cocrystal/cof_cnn_1/bias/v/Read/ReadVariableOpDAdam/deep_cocrystal/interaction_dense_0/kernel/v/Read/ReadVariableOpBAdam/deep_cocrystal/interaction_dense_0/bias/v/Read/ReadVariableOpDAdam/deep_cocrystal/interaction_dense_1/kernel/v/Read/ReadVariableOpBAdam/deep_cocrystal/interaction_dense_1/bias/v/Read/ReadVariableOpDAdam/deep_cocrystal/interaction_dense_2/kernel/v/Read/ReadVariableOpBAdam/deep_cocrystal/interaction_dense_2/bias/v/Read/ReadVariableOpConst_6*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *&
f!R
__inference__traced_save_9314
�
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename#deep_cocrystal/embedding/embeddings%deep_cocrystal/embedding_1/embeddings%deep_cocrystal/prediction_head/kernel#deep_cocrystal/prediction_head/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedeep_cocrystal/api_cnn_0/kerneldeep_cocrystal/api_cnn_0/biasdeep_cocrystal/api_cnn_1/kerneldeep_cocrystal/api_cnn_1/biasdeep_cocrystal/cof_cnn_0/kerneldeep_cocrystal/cof_cnn_0/biasdeep_cocrystal/cof_cnn_1/kerneldeep_cocrystal/cof_cnn_1/bias)deep_cocrystal/interaction_dense_0/kernel'deep_cocrystal/interaction_dense_0/bias)deep_cocrystal/interaction_dense_1/kernel'deep_cocrystal/interaction_dense_1/bias)deep_cocrystal/interaction_dense_2/kernel'deep_cocrystal/interaction_dense_2/biastotalcounttotal_1count_1*Adam/deep_cocrystal/embedding/embeddings/m,Adam/deep_cocrystal/embedding_1/embeddings/m,Adam/deep_cocrystal/prediction_head/kernel/m*Adam/deep_cocrystal/prediction_head/bias/m&Adam/deep_cocrystal/api_cnn_0/kernel/m$Adam/deep_cocrystal/api_cnn_0/bias/m&Adam/deep_cocrystal/api_cnn_1/kernel/m$Adam/deep_cocrystal/api_cnn_1/bias/m&Adam/deep_cocrystal/cof_cnn_0/kernel/m$Adam/deep_cocrystal/cof_cnn_0/bias/m&Adam/deep_cocrystal/cof_cnn_1/kernel/m$Adam/deep_cocrystal/cof_cnn_1/bias/m0Adam/deep_cocrystal/interaction_dense_0/kernel/m.Adam/deep_cocrystal/interaction_dense_0/bias/m0Adam/deep_cocrystal/interaction_dense_1/kernel/m.Adam/deep_cocrystal/interaction_dense_1/bias/m0Adam/deep_cocrystal/interaction_dense_2/kernel/m.Adam/deep_cocrystal/interaction_dense_2/bias/m*Adam/deep_cocrystal/embedding/embeddings/v,Adam/deep_cocrystal/embedding_1/embeddings/v,Adam/deep_cocrystal/prediction_head/kernel/v*Adam/deep_cocrystal/prediction_head/bias/v&Adam/deep_cocrystal/api_cnn_0/kernel/v$Adam/deep_cocrystal/api_cnn_0/bias/v&Adam/deep_cocrystal/api_cnn_1/kernel/v$Adam/deep_cocrystal/api_cnn_1/bias/v&Adam/deep_cocrystal/cof_cnn_0/kernel/v$Adam/deep_cocrystal/cof_cnn_0/bias/v&Adam/deep_cocrystal/cof_cnn_1/kernel/v$Adam/deep_cocrystal/cof_cnn_1/bias/v0Adam/deep_cocrystal/interaction_dense_0/kernel/v.Adam/deep_cocrystal/interaction_dense_0/bias/v0Adam/deep_cocrystal/interaction_dense_1/kernel/v.Adam/deep_cocrystal/interaction_dense_1/bias/v0Adam/deep_cocrystal/interaction_dense_2/kernel/v.Adam/deep_cocrystal/interaction_dense_2/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_restore_9513˓
�
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_9021

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_<lambda>_9074!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identity��,text_file_init/InitializeTableFromTextFileV2�
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_index���������*
offset*
value_index���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
��
�"
__inference__traced_save_9314
file_prefixB
>savev2_deep_cocrystal_embedding_embeddings_read_readvariableopD
@savev2_deep_cocrystal_embedding_1_embeddings_read_readvariableopD
@savev2_deep_cocrystal_prediction_head_kernel_read_readvariableopB
>savev2_deep_cocrystal_prediction_head_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop>
:savev2_deep_cocrystal_api_cnn_0_kernel_read_readvariableop<
8savev2_deep_cocrystal_api_cnn_0_bias_read_readvariableop>
:savev2_deep_cocrystal_api_cnn_1_kernel_read_readvariableop<
8savev2_deep_cocrystal_api_cnn_1_bias_read_readvariableop>
:savev2_deep_cocrystal_cof_cnn_0_kernel_read_readvariableop<
8savev2_deep_cocrystal_cof_cnn_0_bias_read_readvariableop>
:savev2_deep_cocrystal_cof_cnn_1_kernel_read_readvariableop<
8savev2_deep_cocrystal_cof_cnn_1_bias_read_readvariableopH
Dsavev2_deep_cocrystal_interaction_dense_0_kernel_read_readvariableopF
Bsavev2_deep_cocrystal_interaction_dense_0_bias_read_readvariableopH
Dsavev2_deep_cocrystal_interaction_dense_1_kernel_read_readvariableopF
Bsavev2_deep_cocrystal_interaction_dense_1_bias_read_readvariableopH
Dsavev2_deep_cocrystal_interaction_dense_2_kernel_read_readvariableopF
Bsavev2_deep_cocrystal_interaction_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopI
Esavev2_adam_deep_cocrystal_embedding_embeddings_m_read_readvariableopK
Gsavev2_adam_deep_cocrystal_embedding_1_embeddings_m_read_readvariableopK
Gsavev2_adam_deep_cocrystal_prediction_head_kernel_m_read_readvariableopI
Esavev2_adam_deep_cocrystal_prediction_head_bias_m_read_readvariableopE
Asavev2_adam_deep_cocrystal_api_cnn_0_kernel_m_read_readvariableopC
?savev2_adam_deep_cocrystal_api_cnn_0_bias_m_read_readvariableopE
Asavev2_adam_deep_cocrystal_api_cnn_1_kernel_m_read_readvariableopC
?savev2_adam_deep_cocrystal_api_cnn_1_bias_m_read_readvariableopE
Asavev2_adam_deep_cocrystal_cof_cnn_0_kernel_m_read_readvariableopC
?savev2_adam_deep_cocrystal_cof_cnn_0_bias_m_read_readvariableopE
Asavev2_adam_deep_cocrystal_cof_cnn_1_kernel_m_read_readvariableopC
?savev2_adam_deep_cocrystal_cof_cnn_1_bias_m_read_readvariableopO
Ksavev2_adam_deep_cocrystal_interaction_dense_0_kernel_m_read_readvariableopM
Isavev2_adam_deep_cocrystal_interaction_dense_0_bias_m_read_readvariableopO
Ksavev2_adam_deep_cocrystal_interaction_dense_1_kernel_m_read_readvariableopM
Isavev2_adam_deep_cocrystal_interaction_dense_1_bias_m_read_readvariableopO
Ksavev2_adam_deep_cocrystal_interaction_dense_2_kernel_m_read_readvariableopM
Isavev2_adam_deep_cocrystal_interaction_dense_2_bias_m_read_readvariableopI
Esavev2_adam_deep_cocrystal_embedding_embeddings_v_read_readvariableopK
Gsavev2_adam_deep_cocrystal_embedding_1_embeddings_v_read_readvariableopK
Gsavev2_adam_deep_cocrystal_prediction_head_kernel_v_read_readvariableopI
Esavev2_adam_deep_cocrystal_prediction_head_bias_v_read_readvariableopE
Asavev2_adam_deep_cocrystal_api_cnn_0_kernel_v_read_readvariableopC
?savev2_adam_deep_cocrystal_api_cnn_0_bias_v_read_readvariableopE
Asavev2_adam_deep_cocrystal_api_cnn_1_kernel_v_read_readvariableopC
?savev2_adam_deep_cocrystal_api_cnn_1_bias_v_read_readvariableopE
Asavev2_adam_deep_cocrystal_cof_cnn_0_kernel_v_read_readvariableopC
?savev2_adam_deep_cocrystal_cof_cnn_0_bias_v_read_readvariableopE
Asavev2_adam_deep_cocrystal_cof_cnn_1_kernel_v_read_readvariableopC
?savev2_adam_deep_cocrystal_cof_cnn_1_bias_v_read_readvariableopO
Ksavev2_adam_deep_cocrystal_interaction_dense_0_kernel_v_read_readvariableopM
Isavev2_adam_deep_cocrystal_interaction_dense_0_bias_v_read_readvariableopO
Ksavev2_adam_deep_cocrystal_interaction_dense_1_kernel_v_read_readvariableopM
Isavev2_adam_deep_cocrystal_interaction_dense_1_bias_v_read_readvariableopO
Ksavev2_adam_deep_cocrystal_interaction_dense_2_kernel_v_read_readvariableopM
Isavev2_adam_deep_cocrystal_interaction_dense_2_bias_v_read_readvariableop
savev2_const_6

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B3api_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB8coformer_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB1prediction_head/kernel/.ATTRIBUTES/VARIABLE_VALUEB/prediction_head/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBOapi_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTcoformer_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMprediction_head/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKprediction_head/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBOapi_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTcoformer_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMprediction_head/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKprediction_head/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �!
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_deep_cocrystal_embedding_embeddings_read_readvariableop@savev2_deep_cocrystal_embedding_1_embeddings_read_readvariableop@savev2_deep_cocrystal_prediction_head_kernel_read_readvariableop>savev2_deep_cocrystal_prediction_head_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop:savev2_deep_cocrystal_api_cnn_0_kernel_read_readvariableop8savev2_deep_cocrystal_api_cnn_0_bias_read_readvariableop:savev2_deep_cocrystal_api_cnn_1_kernel_read_readvariableop8savev2_deep_cocrystal_api_cnn_1_bias_read_readvariableop:savev2_deep_cocrystal_cof_cnn_0_kernel_read_readvariableop8savev2_deep_cocrystal_cof_cnn_0_bias_read_readvariableop:savev2_deep_cocrystal_cof_cnn_1_kernel_read_readvariableop8savev2_deep_cocrystal_cof_cnn_1_bias_read_readvariableopDsavev2_deep_cocrystal_interaction_dense_0_kernel_read_readvariableopBsavev2_deep_cocrystal_interaction_dense_0_bias_read_readvariableopDsavev2_deep_cocrystal_interaction_dense_1_kernel_read_readvariableopBsavev2_deep_cocrystal_interaction_dense_1_bias_read_readvariableopDsavev2_deep_cocrystal_interaction_dense_2_kernel_read_readvariableopBsavev2_deep_cocrystal_interaction_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopEsavev2_adam_deep_cocrystal_embedding_embeddings_m_read_readvariableopGsavev2_adam_deep_cocrystal_embedding_1_embeddings_m_read_readvariableopGsavev2_adam_deep_cocrystal_prediction_head_kernel_m_read_readvariableopEsavev2_adam_deep_cocrystal_prediction_head_bias_m_read_readvariableopAsavev2_adam_deep_cocrystal_api_cnn_0_kernel_m_read_readvariableop?savev2_adam_deep_cocrystal_api_cnn_0_bias_m_read_readvariableopAsavev2_adam_deep_cocrystal_api_cnn_1_kernel_m_read_readvariableop?savev2_adam_deep_cocrystal_api_cnn_1_bias_m_read_readvariableopAsavev2_adam_deep_cocrystal_cof_cnn_0_kernel_m_read_readvariableop?savev2_adam_deep_cocrystal_cof_cnn_0_bias_m_read_readvariableopAsavev2_adam_deep_cocrystal_cof_cnn_1_kernel_m_read_readvariableop?savev2_adam_deep_cocrystal_cof_cnn_1_bias_m_read_readvariableopKsavev2_adam_deep_cocrystal_interaction_dense_0_kernel_m_read_readvariableopIsavev2_adam_deep_cocrystal_interaction_dense_0_bias_m_read_readvariableopKsavev2_adam_deep_cocrystal_interaction_dense_1_kernel_m_read_readvariableopIsavev2_adam_deep_cocrystal_interaction_dense_1_bias_m_read_readvariableopKsavev2_adam_deep_cocrystal_interaction_dense_2_kernel_m_read_readvariableopIsavev2_adam_deep_cocrystal_interaction_dense_2_bias_m_read_readvariableopEsavev2_adam_deep_cocrystal_embedding_embeddings_v_read_readvariableopGsavev2_adam_deep_cocrystal_embedding_1_embeddings_v_read_readvariableopGsavev2_adam_deep_cocrystal_prediction_head_kernel_v_read_readvariableopEsavev2_adam_deep_cocrystal_prediction_head_bias_v_read_readvariableopAsavev2_adam_deep_cocrystal_api_cnn_0_kernel_v_read_readvariableop?savev2_adam_deep_cocrystal_api_cnn_0_bias_v_read_readvariableopAsavev2_adam_deep_cocrystal_api_cnn_1_kernel_v_read_readvariableop?savev2_adam_deep_cocrystal_api_cnn_1_bias_v_read_readvariableopAsavev2_adam_deep_cocrystal_cof_cnn_0_kernel_v_read_readvariableop?savev2_adam_deep_cocrystal_cof_cnn_0_bias_v_read_readvariableopAsavev2_adam_deep_cocrystal_cof_cnn_1_kernel_v_read_readvariableop?savev2_adam_deep_cocrystal_cof_cnn_1_bias_v_read_readvariableopKsavev2_adam_deep_cocrystal_interaction_dense_0_kernel_v_read_readvariableopIsavev2_adam_deep_cocrystal_interaction_dense_0_bias_v_read_readvariableopKsavev2_adam_deep_cocrystal_interaction_dense_1_kernel_v_read_readvariableopIsavev2_adam_deep_cocrystal_interaction_dense_1_bias_v_read_readvariableopKsavev2_adam_deep_cocrystal_interaction_dense_2_kernel_v_read_readvariableopIsavev2_adam_deep_cocrystal_interaction_dense_2_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :! :! :	�:: : : : : : �:�:��:�: �:�:��:�:
��:�:
��:�:
��:�: : : : :! :! :	�:: �:�:��:�: �:�:��:�:
��:�:
��:�:
��:�:! :! :	�:: �:�:��:�: �:�:��:�:
��:�:
��:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:! :$ 

_output_shapes

:! :%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :)
%
#
_output_shapes
: �:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:)%
#
_output_shapes
: �:!

_output_shapes	
:�:*&
$
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:! :$ 

_output_shapes

:! :%!

_output_shapes
:	�: 

_output_shapes
::) %
#
_output_shapes
: �:!!

_output_shapes	
:�:*"&
$
_output_shapes
:��:!#

_output_shapes	
:�:)$%
#
_output_shapes
: �:!%

_output_shapes	
:�:*&&
$
_output_shapes
:��:!'

_output_shapes	
:�:&("
 
_output_shapes
:
��:!)

_output_shapes	
:�:&*"
 
_output_shapes
:
��:!+

_output_shapes	
:�:&,"
 
_output_shapes
:
��:!-

_output_shapes	
:�:$. 

_output_shapes

:! :$/ 

_output_shapes

:! :%0!

_output_shapes
:	�: 1

_output_shapes
::)2%
#
_output_shapes
: �:!3

_output_shapes	
:�:*4&
$
_output_shapes
:��:!5

_output_shapes	
:�:)6%
#
_output_shapes
: �:!7

_output_shapes	
:�:*8&
$
_output_shapes
:��:!9

_output_shapes	
:�:&:"
 
_output_shapes
:
��:!;

_output_shapes	
:�:&<"
 
_output_shapes
:
��:!=

_output_shapes	
:�:&>"
 
_output_shapes
:
��:!?

_output_shapes	
:�:@

_output_shapes
: 
�
�
E__inference_embedding_1_layer_call_and_return_conditional_losses_8715

inputs	'
embedding_lookup_8709:! 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_8709inputs*
Tindices0	*(
_class
loc:@embedding_lookup/8709*+
_output_shapes
:���������P *
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/8709*+
_output_shapes
:���������P �
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������P Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������P: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_8959

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
+
__inference__destroyer_9050
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_8986

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_7208

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8468
inputs_0
inputs_1O
Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(api_smiles_encoder_string_lookup_equal_y/
+api_smiles_encoder_string_lookup_selectv2_t	Q
Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*cof_smiles_encoder_string_lookup_1_equal_y1
-cof_smiles_encoder_string_lookup_1_selectv2_t	1
embedding_embedding_lookup_8368:! 3
!embedding_1_embedding_lookup_8375:! L
5api_cnn_0_conv1d_expanddims_1_readvariableop_resource: �8
)api_cnn_0_biasadd_readvariableop_resource:	�L
5cof_cnn_0_conv1d_expanddims_1_readvariableop_resource: �8
)cof_cnn_0_biasadd_readvariableop_resource:	�M
5api_cnn_1_conv1d_expanddims_1_readvariableop_resource:��8
)api_cnn_1_biasadd_readvariableop_resource:	�M
5cof_cnn_1_conv1d_expanddims_1_readvariableop_resource:��8
)cof_cnn_1_biasadd_readvariableop_resource:	�F
2interaction_dense_0_matmul_readvariableop_resource:
��B
3interaction_dense_0_biasadd_readvariableop_resource:	�F
2interaction_dense_1_matmul_readvariableop_resource:
��B
3interaction_dense_1_biasadd_readvariableop_resource:	�F
2interaction_dense_2_matmul_readvariableop_resource:
��B
3interaction_dense_2_biasadd_readvariableop_resource:	�A
.prediction_head_matmul_readvariableop_resource:	�=
/prediction_head_biasadd_readvariableop_resource:
identity�� api_cnn_0/BiasAdd/ReadVariableOp�,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp� api_cnn_1/BiasAdd/ReadVariableOp�,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp�>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2� cof_cnn_0/BiasAdd/ReadVariableOp�,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp� cof_cnn_1/BiasAdd/ReadVariableOp�,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp�@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�embedding/embedding_lookup�embedding_1/embedding_lookup�*interaction_dense_0/BiasAdd/ReadVariableOp�)interaction_dense_0/MatMul/ReadVariableOp�*interaction_dense_1/BiasAdd/ReadVariableOp�)interaction_dense_1/MatMul/ReadVariableOp�*interaction_dense_2/BiasAdd/ReadVariableOp�)interaction_dense_2/MatMul/ReadVariableOp�&prediction_head/BiasAdd/ReadVariableOp�%prediction_head/MatMul/ReadVariableOpe
$api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs_0-api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,api_smiles_encoder/StringSplit/strided_sliceStridedSlice6api_smiles_encoder/StringSplit/StringSplitV2:indices:0;api_smiles_encoder/StringSplit/strided_slice/stack:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.api_smiles_encoder/StringSplit/strided_slice_1StridedSlice4api_smiles_encoder/StringSplit/StringSplitV2:shape:0=api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Uapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
capi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasteapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumiapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2iapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle5api_smiles_encoder/StringSplit/StringSplitV2:values:0Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&api_smiles_encoder/string_lookup/EqualEqual5api_smiles_encoder/StringSplit/StringSplitV2:values:0(api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/SelectV2SelectV2*api_smiles_encoder/string_lookup/Equal:z:0+api_smiles_encoder_string_lookup_selectv2_tGapi_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/IdentityIdentity2api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0api_smiles_encoder/RaggedToTensor/Const:output:02api_smiles_encoder/string_lookup/Identity:output:08api_smiles_encoder/RaggedToTensor/default_value:output:07api_smiles_encoder/StringSplit/strided_slice_1:output:05api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSe
$cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs_1-cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,cof_smiles_encoder/StringSplit/strided_sliceStridedSlice6cof_smiles_encoder/StringSplit/StringSplitV2:indices:0;cof_smiles_encoder/StringSplit/strided_slice/stack:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.cof_smiles_encoder/StringSplit/strided_slice_1StridedSlice4cof_smiles_encoder/StringSplit/StringSplitV2:shape:0=cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Ucof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ccof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumicof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2icof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5cof_smiles_encoder/StringSplit/StringSplitV2:values:0Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
(cof_smiles_encoder/string_lookup_1/EqualEqual5cof_smiles_encoder/StringSplit/StringSplitV2:values:0*cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/SelectV2SelectV2,cof_smiles_encoder/string_lookup_1/Equal:z:0-cof_smiles_encoder_string_lookup_1_selectv2_tIcof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/IdentityIdentity4cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0cof_smiles_encoder/RaggedToTensor/Const:output:04cof_smiles_encoder/string_lookup_1/Identity:output:08cof_smiles_encoder/RaggedToTensor/default_value:output:07cof_smiles_encoder/StringSplit/strided_slice_1:output:05cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_8368?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/8368*+
_output_shapes
:���������P *
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/8368*+
_output_shapes
:���������P �
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P V
embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding/NotEqualNotEqual?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_8375?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding_1/embedding_lookup/8375*+
_output_shapes
:���������P *
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/8375*+
_output_shapes
:���������P �
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqual?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������Pj
api_cnn_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
api_cnn_0/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0(api_cnn_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5api_cnn_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0c
!api_cnn_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
api_cnn_0/Conv1D/ExpandDims_1
ExpandDims4api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0*api_cnn_0/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
api_cnn_0/Conv1DConv2D$api_cnn_0/Conv1D/ExpandDims:output:0&api_cnn_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
api_cnn_0/Conv1D/SqueezeSqueezeapi_cnn_0/Conv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

����������
 api_cnn_0/BiasAdd/ReadVariableOpReadVariableOp)api_cnn_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
api_cnn_0/BiasAddBiasAdd!api_cnn_0/Conv1D/Squeeze:output:0(api_cnn_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�i
api_cnn_0/SeluSeluapi_cnn_0/BiasAdd:output:0*
T0*,
_output_shapes
:���������M�j
cof_cnn_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
cof_cnn_0/Conv1D/ExpandDims
ExpandDims0embedding_1/embedding_lookup/Identity_1:output:0(cof_cnn_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5cof_cnn_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0c
!cof_cnn_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
cof_cnn_0/Conv1D/ExpandDims_1
ExpandDims4cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0*cof_cnn_0/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
cof_cnn_0/Conv1DConv2D$cof_cnn_0/Conv1D/ExpandDims:output:0&cof_cnn_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
cof_cnn_0/Conv1D/SqueezeSqueezecof_cnn_0/Conv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

����������
 cof_cnn_0/BiasAdd/ReadVariableOpReadVariableOp)cof_cnn_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cof_cnn_0/BiasAddBiasAdd!cof_cnn_0/Conv1D/Squeeze:output:0(cof_cnn_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�i
cof_cnn_0/SeluSelucof_cnn_0/BiasAdd:output:0*
T0*,
_output_shapes
:���������M�j
api_cnn_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
api_cnn_1/Conv1D/ExpandDims
ExpandDimsapi_cnn_0/Selu:activations:0(api_cnn_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5api_cnn_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0c
!api_cnn_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
api_cnn_1/Conv1D/ExpandDims_1
ExpandDims4api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*api_cnn_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
api_cnn_1/Conv1DConv2D$api_cnn_1/Conv1D/ExpandDims:output:0&api_cnn_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
api_cnn_1/Conv1D/SqueezeSqueezeapi_cnn_1/Conv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

����������
 api_cnn_1/BiasAdd/ReadVariableOpReadVariableOp)api_cnn_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
api_cnn_1/BiasAddBiasAdd!api_cnn_1/Conv1D/Squeeze:output:0(api_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�i
api_cnn_1/SeluSeluapi_cnn_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������J�j
cof_cnn_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
cof_cnn_1/Conv1D/ExpandDims
ExpandDimscof_cnn_0/Selu:activations:0(cof_cnn_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5cof_cnn_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0c
!cof_cnn_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
cof_cnn_1/Conv1D/ExpandDims_1
ExpandDims4cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*cof_cnn_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
cof_cnn_1/Conv1DConv2D$cof_cnn_1/Conv1D/ExpandDims:output:0&cof_cnn_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
cof_cnn_1/Conv1D/SqueezeSqueezecof_cnn_1/Conv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

����������
 cof_cnn_1/BiasAdd/ReadVariableOpReadVariableOp)cof_cnn_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cof_cnn_1/BiasAddBiasAdd!cof_cnn_1/Conv1D/Squeeze:output:0(cof_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�i
cof_cnn_1/SeluSelucof_cnn_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������J�l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_max_pooling1d/MaxMaxapi_cnn_1/Selu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������n
,global_max_pooling1d_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_max_pooling1d_1/MaxMaxcof_cnn_1/Selu:activations:05global_max_pooling1d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d_1/Max:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
)interaction_dense_0/MatMul/ReadVariableOpReadVariableOp2interaction_dense_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
interaction_dense_0/MatMulMatMulconcatenate/concat:output:01interaction_dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*interaction_dense_0/BiasAdd/ReadVariableOpReadVariableOp3interaction_dense_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
interaction_dense_0/BiasAddBiasAdd$interaction_dense_0/MatMul:product:02interaction_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
interaction_dense_0/ReluRelu$interaction_dense_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������w
dropout/IdentityIdentity&interaction_dense_0/Relu:activations:0*
T0*(
_output_shapes
:�����������
)interaction_dense_1/MatMul/ReadVariableOpReadVariableOp2interaction_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
interaction_dense_1/MatMulMatMuldropout/Identity:output:01interaction_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*interaction_dense_1/BiasAdd/ReadVariableOpReadVariableOp3interaction_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
interaction_dense_1/BiasAddBiasAdd$interaction_dense_1/MatMul:product:02interaction_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
interaction_dense_1/ReluRelu$interaction_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������y
dropout_1/IdentityIdentity&interaction_dense_1/Relu:activations:0*
T0*(
_output_shapes
:�����������
)interaction_dense_2/MatMul/ReadVariableOpReadVariableOp2interaction_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
interaction_dense_2/MatMulMatMuldropout_1/Identity:output:01interaction_dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*interaction_dense_2/BiasAdd/ReadVariableOpReadVariableOp3interaction_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
interaction_dense_2/BiasAddBiasAdd$interaction_dense_2/MatMul:product:02interaction_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
interaction_dense_2/ReluRelu$interaction_dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������y
dropout_2/IdentityIdentity&interaction_dense_2/Relu:activations:0*
T0*(
_output_shapes
:�����������
%prediction_head/MatMul/ReadVariableOpReadVariableOp.prediction_head_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
prediction_head/MatMulMatMuldropout_2/Identity:output:0-prediction_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&prediction_head/BiasAdd/ReadVariableOpReadVariableOp/prediction_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
prediction_head/BiasAddBiasAdd prediction_head/MatMul:product:0.prediction_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
prediction_head/SigmoidSigmoid prediction_head/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentityprediction_head/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^api_cnn_0/BiasAdd/ReadVariableOp-^api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp!^api_cnn_1/BiasAdd/ReadVariableOp-^api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp?^api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2!^cof_cnn_0/BiasAdd/ReadVariableOp-^cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp!^cof_cnn_1/BiasAdd/ReadVariableOp-^cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpA^cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2^embedding/embedding_lookup^embedding_1/embedding_lookup+^interaction_dense_0/BiasAdd/ReadVariableOp*^interaction_dense_0/MatMul/ReadVariableOp+^interaction_dense_1/BiasAdd/ReadVariableOp*^interaction_dense_1/MatMul/ReadVariableOp+^interaction_dense_2/BiasAdd/ReadVariableOp*^interaction_dense_2/MatMul/ReadVariableOp'^prediction_head/BiasAdd/ReadVariableOp&^prediction_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 api_cnn_0/BiasAdd/ReadVariableOp api_cnn_0/BiasAdd/ReadVariableOp2\
,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp2D
 api_cnn_1/BiasAdd/ReadVariableOp api_cnn_1/BiasAdd/ReadVariableOp2\
,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp2�
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22D
 cof_cnn_0/BiasAdd/ReadVariableOp cof_cnn_0/BiasAdd/ReadVariableOp2\
,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp2D
 cof_cnn_1/BiasAdd/ReadVariableOp cof_cnn_1/BiasAdd/ReadVariableOp2\
,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp2�
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV228
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2X
*interaction_dense_0/BiasAdd/ReadVariableOp*interaction_dense_0/BiasAdd/ReadVariableOp2V
)interaction_dense_0/MatMul/ReadVariableOp)interaction_dense_0/MatMul/ReadVariableOp2X
*interaction_dense_1/BiasAdd/ReadVariableOp*interaction_dense_1/BiasAdd/ReadVariableOp2V
)interaction_dense_1/MatMul/ReadVariableOp)interaction_dense_1/MatMul/ReadVariableOp2X
*interaction_dense_2/BiasAdd/ReadVariableOp*interaction_dense_2/BiasAdd/ReadVariableOp2V
)interaction_dense_2/MatMul/ReadVariableOp)interaction_dense_2/MatMul/ReadVariableOp2P
&prediction_head/BiasAdd/ReadVariableOp&prediction_head/BiasAdd/ReadVariableOp2N
%prediction_head/MatMul/ReadVariableOp%prediction_head/MatMul/ReadVariableOp:M I
#
_output_shapes
:���������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
_
&__inference_dropout_layer_call_fn_8922

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7379p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
E__inference_concatenate_layer_call_and_return_conditional_losses_8772
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
9
__inference__creator_9055
identity��
hash_table�

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*9
shared_name*(hash_table_./data/vocabulary.txt_-2_-1_2*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�
�
(__inference_api_cnn_1_layer_call_fn_8826

inputs
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_7087t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������J�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������M�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������M�
 
_user_specified_nameinputs
�
Q
5__inference_global_max_pooling1d_1_layer_call_fn_8742

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_6895i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
+
__inference__destroyer_9067
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
�
�
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_7065

inputsB
+conv1d_expanddims_1_readvariableop_resource: �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������M�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������M��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������P 
 
_user_specified_nameinputs
�
�
(__inference_cof_cnn_1_layer_call_fn_8876

inputs
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_7109t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������J�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������M�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������M�
 
_user_specified_nameinputs
�

�
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_7173

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_deep_cocrystal_layer_call_fn_7283
input_1
input_2
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7:! 
	unknown_8:!  
	unknown_9: �

unknown_10:	�!

unknown_11: �

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*'
Tin 
2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7228o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:LH
#
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_8974

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8092
input_1
input_2O
Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(api_smiles_encoder_string_lookup_equal_y/
+api_smiles_encoder_string_lookup_selectv2_t	Q
Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*cof_smiles_encoder_string_lookup_1_equal_y1
-cof_smiles_encoder_string_lookup_1_selectv2_t	 
embedding_8035:! "
embedding_1_8040:! %
api_cnn_0_8045: �
api_cnn_0_8047:	�%
cof_cnn_0_8050: �
cof_cnn_0_8052:	�&
api_cnn_1_8055:��
api_cnn_1_8057:	�&
cof_cnn_1_8060:��
cof_cnn_1_8062:	�,
interaction_dense_0_8068:
��'
interaction_dense_0_8070:	�,
interaction_dense_1_8074:
��'
interaction_dense_1_8076:	�,
interaction_dense_2_8080:
��'
interaction_dense_2_8082:	�'
prediction_head_8086:	�"
prediction_head_8088:
identity��!api_cnn_0/StatefulPartitionedCall�!api_cnn_1/StatefulPartitionedCall�>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2�!cof_cnn_0/StatefulPartitionedCall�!cof_cnn_1/StatefulPartitionedCall�@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�+interaction_dense_0/StatefulPartitionedCall�+interaction_dense_1/StatefulPartitionedCall�+interaction_dense_2/StatefulPartitionedCall�'prediction_head/StatefulPartitionedCalle
$api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2input_1-api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,api_smiles_encoder/StringSplit/strided_sliceStridedSlice6api_smiles_encoder/StringSplit/StringSplitV2:indices:0;api_smiles_encoder/StringSplit/strided_slice/stack:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.api_smiles_encoder/StringSplit/strided_slice_1StridedSlice4api_smiles_encoder/StringSplit/StringSplitV2:shape:0=api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Uapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
capi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasteapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumiapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2iapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle5api_smiles_encoder/StringSplit/StringSplitV2:values:0Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&api_smiles_encoder/string_lookup/EqualEqual5api_smiles_encoder/StringSplit/StringSplitV2:values:0(api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/SelectV2SelectV2*api_smiles_encoder/string_lookup/Equal:z:0+api_smiles_encoder_string_lookup_selectv2_tGapi_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/IdentityIdentity2api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0api_smiles_encoder/RaggedToTensor/Const:output:02api_smiles_encoder/string_lookup/Identity:output:08api_smiles_encoder/RaggedToTensor/default_value:output:07api_smiles_encoder/StringSplit/strided_slice_1:output:05api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSe
$cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2input_2-cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,cof_smiles_encoder/StringSplit/strided_sliceStridedSlice6cof_smiles_encoder/StringSplit/StringSplitV2:indices:0;cof_smiles_encoder/StringSplit/strided_slice/stack:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.cof_smiles_encoder/StringSplit/strided_slice_1StridedSlice4cof_smiles_encoder/StringSplit/StringSplitV2:shape:0=cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Ucof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ccof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumicof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2icof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5cof_smiles_encoder/StringSplit/StringSplitV2:values:0Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
(cof_smiles_encoder/string_lookup_1/EqualEqual5cof_smiles_encoder/StringSplit/StringSplitV2:values:0*cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/SelectV2SelectV2,cof_smiles_encoder/string_lookup_1/Equal:z:0-cof_smiles_encoder_string_lookup_1_selectv2_tIcof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/IdentityIdentity4cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0cof_smiles_encoder/RaggedToTensor/Const:output:04cof_smiles_encoder/string_lookup_1/Identity:output:08cof_smiles_encoder/RaggedToTensor/default_value:output:07cof_smiles_encoder/StringSplit/strided_slice_1:output:05cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_8035*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_7006V
embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding/NotEqualNotEqual?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_8040*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7021X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqual?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
!api_cnn_0/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0api_cnn_0_8045api_cnn_0_8047*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_7043�
!cof_cnn_0/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0cof_cnn_0_8050cof_cnn_0_8052*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_7065�
!api_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*api_cnn_0/StatefulPartitionedCall:output:0api_cnn_1_8055api_cnn_1_8057*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_7087�
!cof_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*cof_cnn_0/StatefulPartitionedCall:output:0cof_cnn_1_8060cof_cnn_1_8062*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_7109�
$global_max_pooling1d/PartitionedCallPartitionedCall*api_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7120�
&global_max_pooling1d_1/PartitionedCallPartitionedCall*cof_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_7127�
concatenate/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7136�
+interaction_dense_0/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0interaction_dense_0_8068interaction_dense_0_8070*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_7149�
dropout/StatefulPartitionedCallStatefulPartitionedCall4interaction_dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7379�
+interaction_dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0interaction_dense_1_8074interaction_dense_1_8076*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_7173�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall4interaction_dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7346�
+interaction_dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0interaction_dense_2_8080interaction_dense_2_8082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_7197�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall4interaction_dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7313�
'prediction_head/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0prediction_head_8086prediction_head_8088*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_prediction_head_layer_call_and_return_conditional_losses_7221
IdentityIdentity0prediction_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^api_cnn_0/StatefulPartitionedCall"^api_cnn_1/StatefulPartitionedCall?^api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2"^cof_cnn_0/StatefulPartitionedCall"^cof_cnn_1/StatefulPartitionedCallA^cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall,^interaction_dense_0/StatefulPartitionedCall,^interaction_dense_1/StatefulPartitionedCall,^interaction_dense_2/StatefulPartitionedCall(^prediction_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!api_cnn_0/StatefulPartitionedCall!api_cnn_0/StatefulPartitionedCall2F
!api_cnn_1/StatefulPartitionedCall!api_cnn_1/StatefulPartitionedCall2�
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22F
!cof_cnn_0/StatefulPartitionedCall!cof_cnn_0/StatefulPartitionedCall2F
!cof_cnn_1/StatefulPartitionedCall!cof_cnn_1/StatefulPartitionedCall2�
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV22B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+interaction_dense_0/StatefulPartitionedCall+interaction_dense_0/StatefulPartitionedCall2Z
+interaction_dense_1/StatefulPartitionedCall+interaction_dense_1/StatefulPartitionedCall2Z
+interaction_dense_2/StatefulPartitionedCall+interaction_dense_2/StatefulPartitionedCall2R
'prediction_head/StatefulPartitionedCall'prediction_head/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:LH
#
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
~
*__inference_embedding_1_layer_call_fn_8706

inputs	
unknown:! 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7021s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������P: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_8927

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_8939

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference__initializer_9062!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identity��,text_file_init/InitializeTableFromTextFileV2�
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_index���������*
offset*
value_index���������G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
�

b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7346

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
I__inference_prediction_head_layer_call_and_return_conditional_losses_7221

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_deep_cocrystal_layer_call_fn_8216
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7:! 
	unknown_8:!  
	unknown_9: �

unknown_10:	�!

unknown_11: �

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*'
Tin 
2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7228o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_9033

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
(__inference_dropout_2_layer_call_fn_9016

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7313p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_7149

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
O
3__inference_global_max_pooling1d_layer_call_fn_8725

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7120a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������J�:T P
,
_output_shapes
:���������J�
 
_user_specified_nameinputs
�
�
C__inference_embedding_layer_call_and_return_conditional_losses_7006

inputs	'
embedding_lookup_7000:! 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_7000inputs*
Tindices0	*(
_class
loc:@embedding_lookup/7000*+
_output_shapes
:���������P *
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/7000*+
_output_shapes
:���������P �
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������P Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������P: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7677

inputs
inputs_1O
Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(api_smiles_encoder_string_lookup_equal_y/
+api_smiles_encoder_string_lookup_selectv2_t	Q
Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*cof_smiles_encoder_string_lookup_1_equal_y1
-cof_smiles_encoder_string_lookup_1_selectv2_t	 
embedding_7620:! "
embedding_1_7625:! %
api_cnn_0_7630: �
api_cnn_0_7632:	�%
cof_cnn_0_7635: �
cof_cnn_0_7637:	�&
api_cnn_1_7640:��
api_cnn_1_7642:	�&
cof_cnn_1_7645:��
cof_cnn_1_7647:	�,
interaction_dense_0_7653:
��'
interaction_dense_0_7655:	�,
interaction_dense_1_7659:
��'
interaction_dense_1_7661:	�,
interaction_dense_2_7665:
��'
interaction_dense_2_7667:	�'
prediction_head_7671:	�"
prediction_head_7673:
identity��!api_cnn_0/StatefulPartitionedCall�!api_cnn_1/StatefulPartitionedCall�>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2�!cof_cnn_0/StatefulPartitionedCall�!cof_cnn_1/StatefulPartitionedCall�@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�dropout/StatefulPartitionedCall�!dropout_1/StatefulPartitionedCall�!dropout_2/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�+interaction_dense_0/StatefulPartitionedCall�+interaction_dense_1/StatefulPartitionedCall�+interaction_dense_2/StatefulPartitionedCall�'prediction_head/StatefulPartitionedCalle
$api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs-api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,api_smiles_encoder/StringSplit/strided_sliceStridedSlice6api_smiles_encoder/StringSplit/StringSplitV2:indices:0;api_smiles_encoder/StringSplit/strided_slice/stack:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.api_smiles_encoder/StringSplit/strided_slice_1StridedSlice4api_smiles_encoder/StringSplit/StringSplitV2:shape:0=api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Uapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
capi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasteapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumiapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2iapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle5api_smiles_encoder/StringSplit/StringSplitV2:values:0Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&api_smiles_encoder/string_lookup/EqualEqual5api_smiles_encoder/StringSplit/StringSplitV2:values:0(api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/SelectV2SelectV2*api_smiles_encoder/string_lookup/Equal:z:0+api_smiles_encoder_string_lookup_selectv2_tGapi_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/IdentityIdentity2api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0api_smiles_encoder/RaggedToTensor/Const:output:02api_smiles_encoder/string_lookup/Identity:output:08api_smiles_encoder/RaggedToTensor/default_value:output:07api_smiles_encoder/StringSplit/strided_slice_1:output:05api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSe
$cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs_1-cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,cof_smiles_encoder/StringSplit/strided_sliceStridedSlice6cof_smiles_encoder/StringSplit/StringSplitV2:indices:0;cof_smiles_encoder/StringSplit/strided_slice/stack:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.cof_smiles_encoder/StringSplit/strided_slice_1StridedSlice4cof_smiles_encoder/StringSplit/StringSplitV2:shape:0=cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Ucof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ccof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumicof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2icof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5cof_smiles_encoder/StringSplit/StringSplitV2:values:0Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
(cof_smiles_encoder/string_lookup_1/EqualEqual5cof_smiles_encoder/StringSplit/StringSplitV2:values:0*cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/SelectV2SelectV2,cof_smiles_encoder/string_lookup_1/Equal:z:0-cof_smiles_encoder_string_lookup_1_selectv2_tIcof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/IdentityIdentity4cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0cof_smiles_encoder/RaggedToTensor/Const:output:04cof_smiles_encoder/string_lookup_1/Identity:output:08cof_smiles_encoder/RaggedToTensor/default_value:output:07cof_smiles_encoder/StringSplit/strided_slice_1:output:05cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_7620*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_7006V
embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding/NotEqualNotEqual?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_7625*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7021X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqual?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
!api_cnn_0/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0api_cnn_0_7630api_cnn_0_7632*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_7043�
!cof_cnn_0/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0cof_cnn_0_7635cof_cnn_0_7637*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_7065�
!api_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*api_cnn_0/StatefulPartitionedCall:output:0api_cnn_1_7640api_cnn_1_7642*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_7087�
!cof_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*cof_cnn_0/StatefulPartitionedCall:output:0cof_cnn_1_7645cof_cnn_1_7647*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_7109�
$global_max_pooling1d/PartitionedCallPartitionedCall*api_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7120�
&global_max_pooling1d_1/PartitionedCallPartitionedCall*cof_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_7127�
concatenate/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7136�
+interaction_dense_0/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0interaction_dense_0_7653interaction_dense_0_7655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_7149�
dropout/StatefulPartitionedCallStatefulPartitionedCall4interaction_dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7379�
+interaction_dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0interaction_dense_1_7659interaction_dense_1_7661*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_7173�
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall4interaction_dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7346�
+interaction_dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0interaction_dense_2_7665interaction_dense_2_7667*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_7197�
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall4interaction_dense_2/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7313�
'prediction_head/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0prediction_head_7671prediction_head_7673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_prediction_head_layer_call_and_return_conditional_losses_7221
IdentityIdentity0prediction_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^api_cnn_0/StatefulPartitionedCall"^api_cnn_1/StatefulPartitionedCall?^api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2"^cof_cnn_0/StatefulPartitionedCall"^cof_cnn_1/StatefulPartitionedCallA^cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall,^interaction_dense_0/StatefulPartitionedCall,^interaction_dense_1/StatefulPartitionedCall,^interaction_dense_2/StatefulPartitionedCall(^prediction_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!api_cnn_0/StatefulPartitionedCall!api_cnn_0/StatefulPartitionedCall2F
!api_cnn_1/StatefulPartitionedCall!api_cnn_1/StatefulPartitionedCall2�
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22F
!cof_cnn_0/StatefulPartitionedCall!cof_cnn_0/StatefulPartitionedCall2F
!cof_cnn_1/StatefulPartitionedCall!cof_cnn_1/StatefulPartitionedCall2�
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV22B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+interaction_dense_0/StatefulPartitionedCall+interaction_dense_0/StatefulPartitionedCall2Z
+interaction_dense_1/StatefulPartitionedCall+interaction_dense_1/StatefulPartitionedCall2Z
+interaction_dense_2/StatefulPartitionedCall+interaction_dense_2/StatefulPartitionedCall2R
'prediction_head/StatefulPartitionedCall'prediction_head/StatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
��
�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7941
input_1
input_2O
Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(api_smiles_encoder_string_lookup_equal_y/
+api_smiles_encoder_string_lookup_selectv2_t	Q
Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*cof_smiles_encoder_string_lookup_1_equal_y1
-cof_smiles_encoder_string_lookup_1_selectv2_t	 
embedding_7884:! "
embedding_1_7889:! %
api_cnn_0_7894: �
api_cnn_0_7896:	�%
cof_cnn_0_7899: �
cof_cnn_0_7901:	�&
api_cnn_1_7904:��
api_cnn_1_7906:	�&
cof_cnn_1_7909:��
cof_cnn_1_7911:	�,
interaction_dense_0_7917:
��'
interaction_dense_0_7919:	�,
interaction_dense_1_7923:
��'
interaction_dense_1_7925:	�,
interaction_dense_2_7929:
��'
interaction_dense_2_7931:	�'
prediction_head_7935:	�"
prediction_head_7937:
identity��!api_cnn_0/StatefulPartitionedCall�!api_cnn_1/StatefulPartitionedCall�>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2�!cof_cnn_0/StatefulPartitionedCall�!cof_cnn_1/StatefulPartitionedCall�@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�+interaction_dense_0/StatefulPartitionedCall�+interaction_dense_1/StatefulPartitionedCall�+interaction_dense_2/StatefulPartitionedCall�'prediction_head/StatefulPartitionedCalle
$api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2input_1-api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,api_smiles_encoder/StringSplit/strided_sliceStridedSlice6api_smiles_encoder/StringSplit/StringSplitV2:indices:0;api_smiles_encoder/StringSplit/strided_slice/stack:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.api_smiles_encoder/StringSplit/strided_slice_1StridedSlice4api_smiles_encoder/StringSplit/StringSplitV2:shape:0=api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Uapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
capi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasteapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumiapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2iapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle5api_smiles_encoder/StringSplit/StringSplitV2:values:0Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&api_smiles_encoder/string_lookup/EqualEqual5api_smiles_encoder/StringSplit/StringSplitV2:values:0(api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/SelectV2SelectV2*api_smiles_encoder/string_lookup/Equal:z:0+api_smiles_encoder_string_lookup_selectv2_tGapi_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/IdentityIdentity2api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0api_smiles_encoder/RaggedToTensor/Const:output:02api_smiles_encoder/string_lookup/Identity:output:08api_smiles_encoder/RaggedToTensor/default_value:output:07api_smiles_encoder/StringSplit/strided_slice_1:output:05api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSe
$cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2input_2-cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,cof_smiles_encoder/StringSplit/strided_sliceStridedSlice6cof_smiles_encoder/StringSplit/StringSplitV2:indices:0;cof_smiles_encoder/StringSplit/strided_slice/stack:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.cof_smiles_encoder/StringSplit/strided_slice_1StridedSlice4cof_smiles_encoder/StringSplit/StringSplitV2:shape:0=cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Ucof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ccof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumicof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2icof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5cof_smiles_encoder/StringSplit/StringSplitV2:values:0Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
(cof_smiles_encoder/string_lookup_1/EqualEqual5cof_smiles_encoder/StringSplit/StringSplitV2:values:0*cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/SelectV2SelectV2,cof_smiles_encoder/string_lookup_1/Equal:z:0-cof_smiles_encoder_string_lookup_1_selectv2_tIcof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/IdentityIdentity4cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0cof_smiles_encoder/RaggedToTensor/Const:output:04cof_smiles_encoder/string_lookup_1/Identity:output:08cof_smiles_encoder/RaggedToTensor/default_value:output:07cof_smiles_encoder/StringSplit/strided_slice_1:output:05cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_7884*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_7006V
embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding/NotEqualNotEqual?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_7889*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7021X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqual?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
!api_cnn_0/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0api_cnn_0_7894api_cnn_0_7896*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_7043�
!cof_cnn_0/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0cof_cnn_0_7899cof_cnn_0_7901*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_7065�
!api_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*api_cnn_0/StatefulPartitionedCall:output:0api_cnn_1_7904api_cnn_1_7906*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_7087�
!cof_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*cof_cnn_0/StatefulPartitionedCall:output:0cof_cnn_1_7909cof_cnn_1_7911*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_7109�
$global_max_pooling1d/PartitionedCallPartitionedCall*api_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7120�
&global_max_pooling1d_1/PartitionedCallPartitionedCall*cof_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_7127�
concatenate/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7136�
+interaction_dense_0/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0interaction_dense_0_7917interaction_dense_0_7919*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_7149�
dropout/PartitionedCallPartitionedCall4interaction_dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7160�
+interaction_dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0interaction_dense_1_7923interaction_dense_1_7925*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_7173�
dropout_1/PartitionedCallPartitionedCall4interaction_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7184�
+interaction_dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0interaction_dense_2_7929interaction_dense_2_7931*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_7197�
dropout_2/PartitionedCallPartitionedCall4interaction_dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7208�
'prediction_head/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0prediction_head_7935prediction_head_7937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_prediction_head_layer_call_and_return_conditional_losses_7221
IdentityIdentity0prediction_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^api_cnn_0/StatefulPartitionedCall"^api_cnn_1/StatefulPartitionedCall?^api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2"^cof_cnn_0/StatefulPartitionedCall"^cof_cnn_1/StatefulPartitionedCallA^cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall,^interaction_dense_0/StatefulPartitionedCall,^interaction_dense_1/StatefulPartitionedCall,^interaction_dense_2/StatefulPartitionedCall(^prediction_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!api_cnn_0/StatefulPartitionedCall!api_cnn_0/StatefulPartitionedCall2F
!api_cnn_1/StatefulPartitionedCall!api_cnn_1/StatefulPartitionedCall2�
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22F
!cof_cnn_0/StatefulPartitionedCall!cof_cnn_0/StatefulPartitionedCall2F
!cof_cnn_1/StatefulPartitionedCall!cof_cnn_1/StatefulPartitionedCall2�
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV22F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+interaction_dense_0/StatefulPartitionedCall+interaction_dense_0/StatefulPartitionedCall2Z
+interaction_dense_1/StatefulPartitionedCall+interaction_dense_1/StatefulPartitionedCall2Z
+interaction_dense_2/StatefulPartitionedCall+interaction_dense_2/StatefulPartitionedCall2R
'prediction_head/StatefulPartitionedCall'prediction_head/StatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:LH
#
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

`
A__inference_dropout_layer_call_and_return_conditional_losses_7379

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
B
&__inference_dropout_layer_call_fn_8917

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7160a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
|
(__inference_embedding_layer_call_fn_8690

inputs	
unknown:! 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_7006s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������P `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������P: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
E__inference_embedding_1_layer_call_and_return_conditional_losses_7021

inputs	'
embedding_lookup_7015:! 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_7015inputs*
Tindices0	*(
_class
loc:@embedding_lookup/7015*+
_output_shapes
:���������P *
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/7015*+
_output_shapes
:���������P �
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������P Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������P: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_8912

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_7109

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������J�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������J��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������M�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:���������M�
 
_user_specified_nameinputs
�
�
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_7043

inputsB
+conv1d_expanddims_1_readvariableop_resource: �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������M�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������M��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������P 
 
_user_specified_nameinputs
�
j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7120

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:����������U
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������J�:T P
,
_output_shapes
:���������J�
 
_user_specified_nameinputs
�
l
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_7127

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:����������U
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������J�:T P
,
_output_shapes
:���������J�
 
_user_specified_nameinputs
�
Q
5__inference_global_max_pooling1d_1_layer_call_fn_8747

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_7127a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������J�:T P
,
_output_shapes
:���������J�
 
_user_specified_nameinputs
�
�
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_8817

inputsB
+conv1d_expanddims_1_readvariableop_resource: �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������M�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������M��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������P 
 
_user_specified_nameinputs
�
l
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_8759

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:����������U
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������J�:T P
,
_output_shapes
:���������J�
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_7160

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7184

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_8867

inputsB
+conv1d_expanddims_1_readvariableop_resource: �.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������M�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������M��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:���������P 
 
_user_specified_nameinputs
�
D
(__inference_dropout_1_layer_call_fn_8964

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7184a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
__inference__wrapped_model_6872
input_1
input_2^
Zdeep_cocrystal_api_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle_
[deep_cocrystal_api_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	;
7deep_cocrystal_api_smiles_encoder_string_lookup_equal_y>
:deep_cocrystal_api_smiles_encoder_string_lookup_selectv2_t	`
\deep_cocrystal_cof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handlea
]deep_cocrystal_cof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	=
9deep_cocrystal_cof_smiles_encoder_string_lookup_1_equal_y@
<deep_cocrystal_cof_smiles_encoder_string_lookup_1_selectv2_t	@
.deep_cocrystal_embedding_embedding_lookup_6772:! B
0deep_cocrystal_embedding_1_embedding_lookup_6779:! [
Ddeep_cocrystal_api_cnn_0_conv1d_expanddims_1_readvariableop_resource: �G
8deep_cocrystal_api_cnn_0_biasadd_readvariableop_resource:	�[
Ddeep_cocrystal_cof_cnn_0_conv1d_expanddims_1_readvariableop_resource: �G
8deep_cocrystal_cof_cnn_0_biasadd_readvariableop_resource:	�\
Ddeep_cocrystal_api_cnn_1_conv1d_expanddims_1_readvariableop_resource:��G
8deep_cocrystal_api_cnn_1_biasadd_readvariableop_resource:	�\
Ddeep_cocrystal_cof_cnn_1_conv1d_expanddims_1_readvariableop_resource:��G
8deep_cocrystal_cof_cnn_1_biasadd_readvariableop_resource:	�U
Adeep_cocrystal_interaction_dense_0_matmul_readvariableop_resource:
��Q
Bdeep_cocrystal_interaction_dense_0_biasadd_readvariableop_resource:	�U
Adeep_cocrystal_interaction_dense_1_matmul_readvariableop_resource:
��Q
Bdeep_cocrystal_interaction_dense_1_biasadd_readvariableop_resource:	�U
Adeep_cocrystal_interaction_dense_2_matmul_readvariableop_resource:
��Q
Bdeep_cocrystal_interaction_dense_2_biasadd_readvariableop_resource:	�P
=deep_cocrystal_prediction_head_matmul_readvariableop_resource:	�L
>deep_cocrystal_prediction_head_biasadd_readvariableop_resource:
identity��/deep_cocrystal/api_cnn_0/BiasAdd/ReadVariableOp�;deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp�/deep_cocrystal/api_cnn_1/BiasAdd/ReadVariableOp�;deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp�Mdeep_cocrystal/api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2�/deep_cocrystal/cof_cnn_0/BiasAdd/ReadVariableOp�;deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp�/deep_cocrystal/cof_cnn_1/BiasAdd/ReadVariableOp�;deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp�Odeep_cocrystal/cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�)deep_cocrystal/embedding/embedding_lookup�+deep_cocrystal/embedding_1/embedding_lookup�9deep_cocrystal/interaction_dense_0/BiasAdd/ReadVariableOp�8deep_cocrystal/interaction_dense_0/MatMul/ReadVariableOp�9deep_cocrystal/interaction_dense_1/BiasAdd/ReadVariableOp�8deep_cocrystal/interaction_dense_1/MatMul/ReadVariableOp�9deep_cocrystal/interaction_dense_2/BiasAdd/ReadVariableOp�8deep_cocrystal/interaction_dense_2/MatMul/ReadVariableOp�5deep_cocrystal/prediction_head/BiasAdd/ReadVariableOp�4deep_cocrystal/prediction_head/MatMul/ReadVariableOpt
3deep_cocrystal/api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
;deep_cocrystal/api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2input_1<deep_cocrystal/api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
Adeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Cdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
Cdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
;deep_cocrystal/api_smiles_encoder/StringSplit/strided_sliceStridedSliceEdeep_cocrystal/api_smiles_encoder/StringSplit/StringSplitV2:indices:0Jdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice/stack:output:0Ldeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0Ldeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
Cdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Edeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Edeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=deep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1StridedSliceCdeep_cocrystal/api_smiles_encoder/StringSplit/StringSplitV2:shape:0Ldeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0Ndeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0Ndeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
ddeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastDdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
fdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastFdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
ndeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapehdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
ndeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
mdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdwdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0wdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
rdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatervdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0{deep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
mdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasttdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ldeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxhdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ydeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
ndeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ldeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2udeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0wdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ldeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulqdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumjdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumjdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0tdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
pdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
qdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounthdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ydeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
kdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
fdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumxdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0tdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
odeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
kdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
fdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2xdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ldeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0tdeep_cocrystal/api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Mdeep_cocrystal/api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Zdeep_cocrystal_api_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleDdeep_cocrystal/api_smiles_encoder/StringSplit/StringSplitV2:values:0[deep_cocrystal_api_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
5deep_cocrystal/api_smiles_encoder/string_lookup/EqualEqualDdeep_cocrystal/api_smiles_encoder/StringSplit/StringSplitV2:values:07deep_cocrystal_api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
8deep_cocrystal/api_smiles_encoder/string_lookup/SelectV2SelectV29deep_cocrystal/api_smiles_encoder/string_lookup/Equal:z:0:deep_cocrystal_api_smiles_encoder_string_lookup_selectv2_tVdeep_cocrystal/api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
8deep_cocrystal/api_smiles_encoder/string_lookup/IdentityIdentityAdeep_cocrystal/api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:����������
>deep_cocrystal/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
6deep_cocrystal/api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
Edeep_cocrystal/api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor?deep_cocrystal/api_smiles_encoder/RaggedToTensor/Const:output:0Adeep_cocrystal/api_smiles_encoder/string_lookup/Identity:output:0Gdeep_cocrystal/api_smiles_encoder/RaggedToTensor/default_value:output:0Fdeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice_1:output:0Ddeep_cocrystal/api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSt
3deep_cocrystal/cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
;deep_cocrystal/cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2input_2<deep_cocrystal/cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
Adeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
Cdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
Cdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
;deep_cocrystal/cof_smiles_encoder/StringSplit/strided_sliceStridedSliceEdeep_cocrystal/cof_smiles_encoder/StringSplit/StringSplitV2:indices:0Jdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice/stack:output:0Ldeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0Ldeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask�
Cdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
Edeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
Edeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
=deep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1StridedSliceCdeep_cocrystal/cof_smiles_encoder/StringSplit/StringSplitV2:shape:0Ldeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0Ndeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0Ndeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
ddeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCastDdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
fdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1CastFdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
ndeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapehdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
ndeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
mdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdwdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0wdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
rdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatervdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0{deep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
mdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasttdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
ldeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxhdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ydeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
ndeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
ldeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2udeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0wdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
ldeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulqdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximumjdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimumjdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0tdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
pdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
qdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincounthdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0tdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0ydeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
kdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
fdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumxdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0tdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
odeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
kdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
fdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2xdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0ldeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0tdeep_cocrystal/cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
Odeep_cocrystal/cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2\deep_cocrystal_cof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleDdeep_cocrystal/cof_smiles_encoder/StringSplit/StringSplitV2:values:0]deep_cocrystal_cof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
7deep_cocrystal/cof_smiles_encoder/string_lookup_1/EqualEqualDdeep_cocrystal/cof_smiles_encoder/StringSplit/StringSplitV2:values:09deep_cocrystal_cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
:deep_cocrystal/cof_smiles_encoder/string_lookup_1/SelectV2SelectV2;deep_cocrystal/cof_smiles_encoder/string_lookup_1/Equal:z:0<deep_cocrystal_cof_smiles_encoder_string_lookup_1_selectv2_tXdeep_cocrystal/cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
:deep_cocrystal/cof_smiles_encoder/string_lookup_1/IdentityIdentityCdeep_cocrystal/cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:����������
>deep_cocrystal/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
6deep_cocrystal/cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
Edeep_cocrystal/cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor?deep_cocrystal/cof_smiles_encoder/RaggedToTensor/Const:output:0Cdeep_cocrystal/cof_smiles_encoder/string_lookup_1/Identity:output:0Gdeep_cocrystal/cof_smiles_encoder/RaggedToTensor/default_value:output:0Fdeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice_1:output:0Ddeep_cocrystal/cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
)deep_cocrystal/embedding/embedding_lookupResourceGather.deep_cocrystal_embedding_embedding_lookup_6772Ndeep_cocrystal/api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*A
_class7
53loc:@deep_cocrystal/embedding/embedding_lookup/6772*+
_output_shapes
:���������P *
dtype0�
2deep_cocrystal/embedding/embedding_lookup/IdentityIdentity2deep_cocrystal/embedding/embedding_lookup:output:0*
T0*A
_class7
53loc:@deep_cocrystal/embedding/embedding_lookup/6772*+
_output_shapes
:���������P �
4deep_cocrystal/embedding/embedding_lookup/Identity_1Identity;deep_cocrystal/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P e
#deep_cocrystal/embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
!deep_cocrystal/embedding/NotEqualNotEqualNdeep_cocrystal/api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0,deep_cocrystal/embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
+deep_cocrystal/embedding_1/embedding_lookupResourceGather0deep_cocrystal_embedding_1_embedding_lookup_6779Ndeep_cocrystal/cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*C
_class9
75loc:@deep_cocrystal/embedding_1/embedding_lookup/6779*+
_output_shapes
:���������P *
dtype0�
4deep_cocrystal/embedding_1/embedding_lookup/IdentityIdentity4deep_cocrystal/embedding_1/embedding_lookup:output:0*
T0*C
_class9
75loc:@deep_cocrystal/embedding_1/embedding_lookup/6779*+
_output_shapes
:���������P �
6deep_cocrystal/embedding_1/embedding_lookup/Identity_1Identity=deep_cocrystal/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P g
%deep_cocrystal/embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
#deep_cocrystal/embedding_1/NotEqualNotEqualNdeep_cocrystal/cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0.deep_cocrystal/embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������Py
.deep_cocrystal/api_cnn_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
*deep_cocrystal/api_cnn_0/Conv1D/ExpandDims
ExpandDims=deep_cocrystal/embedding/embedding_lookup/Identity_1:output:07deep_cocrystal/api_cnn_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
;deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpDdeep_cocrystal_api_cnn_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0r
0deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
,deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1
ExpandDimsCdeep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp:value:09deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
deep_cocrystal/api_cnn_0/Conv1DConv2D3deep_cocrystal/api_cnn_0/Conv1D/ExpandDims:output:05deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
'deep_cocrystal/api_cnn_0/Conv1D/SqueezeSqueeze(deep_cocrystal/api_cnn_0/Conv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

����������
/deep_cocrystal/api_cnn_0/BiasAdd/ReadVariableOpReadVariableOp8deep_cocrystal_api_cnn_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 deep_cocrystal/api_cnn_0/BiasAddBiasAdd0deep_cocrystal/api_cnn_0/Conv1D/Squeeze:output:07deep_cocrystal/api_cnn_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M��
deep_cocrystal/api_cnn_0/SeluSelu)deep_cocrystal/api_cnn_0/BiasAdd:output:0*
T0*,
_output_shapes
:���������M�y
.deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
*deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims
ExpandDims?deep_cocrystal/embedding_1/embedding_lookup/Identity_1:output:07deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
;deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpDdeep_cocrystal_cof_cnn_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0r
0deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
,deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1
ExpandDimsCdeep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp:value:09deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
deep_cocrystal/cof_cnn_0/Conv1DConv2D3deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims:output:05deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
'deep_cocrystal/cof_cnn_0/Conv1D/SqueezeSqueeze(deep_cocrystal/cof_cnn_0/Conv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

����������
/deep_cocrystal/cof_cnn_0/BiasAdd/ReadVariableOpReadVariableOp8deep_cocrystal_cof_cnn_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 deep_cocrystal/cof_cnn_0/BiasAddBiasAdd0deep_cocrystal/cof_cnn_0/Conv1D/Squeeze:output:07deep_cocrystal/cof_cnn_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M��
deep_cocrystal/cof_cnn_0/SeluSelu)deep_cocrystal/cof_cnn_0/BiasAdd:output:0*
T0*,
_output_shapes
:���������M�y
.deep_cocrystal/api_cnn_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
*deep_cocrystal/api_cnn_1/Conv1D/ExpandDims
ExpandDims+deep_cocrystal/api_cnn_0/Selu:activations:07deep_cocrystal/api_cnn_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
;deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpDdeep_cocrystal_api_cnn_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0r
0deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
,deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1
ExpandDimsCdeep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp:value:09deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
deep_cocrystal/api_cnn_1/Conv1DConv2D3deep_cocrystal/api_cnn_1/Conv1D/ExpandDims:output:05deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
'deep_cocrystal/api_cnn_1/Conv1D/SqueezeSqueeze(deep_cocrystal/api_cnn_1/Conv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

����������
/deep_cocrystal/api_cnn_1/BiasAdd/ReadVariableOpReadVariableOp8deep_cocrystal_api_cnn_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 deep_cocrystal/api_cnn_1/BiasAddBiasAdd0deep_cocrystal/api_cnn_1/Conv1D/Squeeze:output:07deep_cocrystal/api_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J��
deep_cocrystal/api_cnn_1/SeluSelu)deep_cocrystal/api_cnn_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������J�y
.deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
*deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims
ExpandDims+deep_cocrystal/cof_cnn_0/Selu:activations:07deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
;deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpDdeep_cocrystal_cof_cnn_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0r
0deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
,deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1
ExpandDimsCdeep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp:value:09deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
deep_cocrystal/cof_cnn_1/Conv1DConv2D3deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims:output:05deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
'deep_cocrystal/cof_cnn_1/Conv1D/SqueezeSqueeze(deep_cocrystal/cof_cnn_1/Conv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

����������
/deep_cocrystal/cof_cnn_1/BiasAdd/ReadVariableOpReadVariableOp8deep_cocrystal_cof_cnn_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 deep_cocrystal/cof_cnn_1/BiasAddBiasAdd0deep_cocrystal/cof_cnn_1/Conv1D/Squeeze:output:07deep_cocrystal/cof_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J��
deep_cocrystal/cof_cnn_1/SeluSelu)deep_cocrystal/cof_cnn_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������J�{
9deep_cocrystal/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
'deep_cocrystal/global_max_pooling1d/MaxMax+deep_cocrystal/api_cnn_1/Selu:activations:0Bdeep_cocrystal/global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������}
;deep_cocrystal/global_max_pooling1d_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
)deep_cocrystal/global_max_pooling1d_1/MaxMax+deep_cocrystal/cof_cnn_1/Selu:activations:0Ddeep_cocrystal/global_max_pooling1d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������h
&deep_cocrystal/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
!deep_cocrystal/concatenate/concatConcatV20deep_cocrystal/global_max_pooling1d/Max:output:02deep_cocrystal/global_max_pooling1d_1/Max:output:0/deep_cocrystal/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
8deep_cocrystal/interaction_dense_0/MatMul/ReadVariableOpReadVariableOpAdeep_cocrystal_interaction_dense_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
)deep_cocrystal/interaction_dense_0/MatMulMatMul*deep_cocrystal/concatenate/concat:output:0@deep_cocrystal/interaction_dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9deep_cocrystal/interaction_dense_0/BiasAdd/ReadVariableOpReadVariableOpBdeep_cocrystal_interaction_dense_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*deep_cocrystal/interaction_dense_0/BiasAddBiasAdd3deep_cocrystal/interaction_dense_0/MatMul:product:0Adeep_cocrystal/interaction_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'deep_cocrystal/interaction_dense_0/ReluRelu3deep_cocrystal/interaction_dense_0/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
deep_cocrystal/dropout/IdentityIdentity5deep_cocrystal/interaction_dense_0/Relu:activations:0*
T0*(
_output_shapes
:�����������
8deep_cocrystal/interaction_dense_1/MatMul/ReadVariableOpReadVariableOpAdeep_cocrystal_interaction_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
)deep_cocrystal/interaction_dense_1/MatMulMatMul(deep_cocrystal/dropout/Identity:output:0@deep_cocrystal/interaction_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9deep_cocrystal/interaction_dense_1/BiasAdd/ReadVariableOpReadVariableOpBdeep_cocrystal_interaction_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*deep_cocrystal/interaction_dense_1/BiasAddBiasAdd3deep_cocrystal/interaction_dense_1/MatMul:product:0Adeep_cocrystal/interaction_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'deep_cocrystal/interaction_dense_1/ReluRelu3deep_cocrystal/interaction_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!deep_cocrystal/dropout_1/IdentityIdentity5deep_cocrystal/interaction_dense_1/Relu:activations:0*
T0*(
_output_shapes
:�����������
8deep_cocrystal/interaction_dense_2/MatMul/ReadVariableOpReadVariableOpAdeep_cocrystal_interaction_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
)deep_cocrystal/interaction_dense_2/MatMulMatMul*deep_cocrystal/dropout_1/Identity:output:0@deep_cocrystal/interaction_dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
9deep_cocrystal/interaction_dense_2/BiasAdd/ReadVariableOpReadVariableOpBdeep_cocrystal_interaction_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*deep_cocrystal/interaction_dense_2/BiasAddBiasAdd3deep_cocrystal/interaction_dense_2/MatMul:product:0Adeep_cocrystal/interaction_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'deep_cocrystal/interaction_dense_2/ReluRelu3deep_cocrystal/interaction_dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!deep_cocrystal/dropout_2/IdentityIdentity5deep_cocrystal/interaction_dense_2/Relu:activations:0*
T0*(
_output_shapes
:�����������
4deep_cocrystal/prediction_head/MatMul/ReadVariableOpReadVariableOp=deep_cocrystal_prediction_head_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%deep_cocrystal/prediction_head/MatMulMatMul*deep_cocrystal/dropout_2/Identity:output:0<deep_cocrystal/prediction_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5deep_cocrystal/prediction_head/BiasAdd/ReadVariableOpReadVariableOp>deep_cocrystal_prediction_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&deep_cocrystal/prediction_head/BiasAddBiasAdd/deep_cocrystal/prediction_head/MatMul:product:0=deep_cocrystal/prediction_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&deep_cocrystal/prediction_head/SigmoidSigmoid/deep_cocrystal/prediction_head/BiasAdd:output:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*deep_cocrystal/prediction_head/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������	
NoOpNoOp0^deep_cocrystal/api_cnn_0/BiasAdd/ReadVariableOp<^deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp0^deep_cocrystal/api_cnn_1/BiasAdd/ReadVariableOp<^deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpN^deep_cocrystal/api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV20^deep_cocrystal/cof_cnn_0/BiasAdd/ReadVariableOp<^deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp0^deep_cocrystal/cof_cnn_1/BiasAdd/ReadVariableOp<^deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpP^deep_cocrystal/cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2*^deep_cocrystal/embedding/embedding_lookup,^deep_cocrystal/embedding_1/embedding_lookup:^deep_cocrystal/interaction_dense_0/BiasAdd/ReadVariableOp9^deep_cocrystal/interaction_dense_0/MatMul/ReadVariableOp:^deep_cocrystal/interaction_dense_1/BiasAdd/ReadVariableOp9^deep_cocrystal/interaction_dense_1/MatMul/ReadVariableOp:^deep_cocrystal/interaction_dense_2/BiasAdd/ReadVariableOp9^deep_cocrystal/interaction_dense_2/MatMul/ReadVariableOp6^deep_cocrystal/prediction_head/BiasAdd/ReadVariableOp5^deep_cocrystal/prediction_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/deep_cocrystal/api_cnn_0/BiasAdd/ReadVariableOp/deep_cocrystal/api_cnn_0/BiasAdd/ReadVariableOp2z
;deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp;deep_cocrystal/api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp2b
/deep_cocrystal/api_cnn_1/BiasAdd/ReadVariableOp/deep_cocrystal/api_cnn_1/BiasAdd/ReadVariableOp2z
;deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp;deep_cocrystal/api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp2�
Mdeep_cocrystal/api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2Mdeep_cocrystal/api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22b
/deep_cocrystal/cof_cnn_0/BiasAdd/ReadVariableOp/deep_cocrystal/cof_cnn_0/BiasAdd/ReadVariableOp2z
;deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp;deep_cocrystal/cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp2b
/deep_cocrystal/cof_cnn_1/BiasAdd/ReadVariableOp/deep_cocrystal/cof_cnn_1/BiasAdd/ReadVariableOp2z
;deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp;deep_cocrystal/cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp2�
Odeep_cocrystal/cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2Odeep_cocrystal/cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV22V
)deep_cocrystal/embedding/embedding_lookup)deep_cocrystal/embedding/embedding_lookup2Z
+deep_cocrystal/embedding_1/embedding_lookup+deep_cocrystal/embedding_1/embedding_lookup2v
9deep_cocrystal/interaction_dense_0/BiasAdd/ReadVariableOp9deep_cocrystal/interaction_dense_0/BiasAdd/ReadVariableOp2t
8deep_cocrystal/interaction_dense_0/MatMul/ReadVariableOp8deep_cocrystal/interaction_dense_0/MatMul/ReadVariableOp2v
9deep_cocrystal/interaction_dense_1/BiasAdd/ReadVariableOp9deep_cocrystal/interaction_dense_1/BiasAdd/ReadVariableOp2t
8deep_cocrystal/interaction_dense_1/MatMul/ReadVariableOp8deep_cocrystal/interaction_dense_1/MatMul/ReadVariableOp2v
9deep_cocrystal/interaction_dense_2/BiasAdd/ReadVariableOp9deep_cocrystal/interaction_dense_2/BiasAdd/ReadVariableOp2t
8deep_cocrystal/interaction_dense_2/MatMul/ReadVariableOp8deep_cocrystal/interaction_dense_2/MatMul/ReadVariableOp2n
5deep_cocrystal/prediction_head/BiasAdd/ReadVariableOp5deep_cocrystal/prediction_head/BiasAdd/ReadVariableOp2l
4deep_cocrystal/prediction_head/MatMul/ReadVariableOp4deep_cocrystal/prediction_head/MatMul/ReadVariableOp:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:LH
#
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
�
2__inference_interaction_dense_0_layer_call_fn_8901

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_7149p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
V
*__inference_concatenate_layer_call_fn_8765
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7136a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
C__inference_embedding_layer_call_and_return_conditional_losses_8699

inputs	'
embedding_lookup_8693:! 
identity��embedding_lookup�
embedding_lookupResourceGatherembedding_lookup_8693inputs*
Tindices0	*(
_class
loc:@embedding_lookup/8693*+
_output_shapes
:���������P *
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*(
_class
loc:@embedding_lookup/8693*+
_output_shapes
:���������P �
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:���������P Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������P: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
.__inference_prediction_head_layer_call_fn_8781

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_prediction_head_layer_call_and_return_conditional_losses_7221o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7228

inputs
inputs_1O
Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(api_smiles_encoder_string_lookup_equal_y/
+api_smiles_encoder_string_lookup_selectv2_t	Q
Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*cof_smiles_encoder_string_lookup_1_equal_y1
-cof_smiles_encoder_string_lookup_1_selectv2_t	 
embedding_7007:! "
embedding_1_7022:! %
api_cnn_0_7044: �
api_cnn_0_7046:	�%
cof_cnn_0_7066: �
cof_cnn_0_7068:	�&
api_cnn_1_7088:��
api_cnn_1_7090:	�&
cof_cnn_1_7110:��
cof_cnn_1_7112:	�,
interaction_dense_0_7150:
��'
interaction_dense_0_7152:	�,
interaction_dense_1_7174:
��'
interaction_dense_1_7176:	�,
interaction_dense_2_7198:
��'
interaction_dense_2_7200:	�'
prediction_head_7222:	�"
prediction_head_7224:
identity��!api_cnn_0/StatefulPartitionedCall�!api_cnn_1/StatefulPartitionedCall�>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2�!cof_cnn_0/StatefulPartitionedCall�!cof_cnn_1/StatefulPartitionedCall�@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�+interaction_dense_0/StatefulPartitionedCall�+interaction_dense_1/StatefulPartitionedCall�+interaction_dense_2/StatefulPartitionedCall�'prediction_head/StatefulPartitionedCalle
$api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs-api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,api_smiles_encoder/StringSplit/strided_sliceStridedSlice6api_smiles_encoder/StringSplit/StringSplitV2:indices:0;api_smiles_encoder/StringSplit/strided_slice/stack:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.api_smiles_encoder/StringSplit/strided_slice_1StridedSlice4api_smiles_encoder/StringSplit/StringSplitV2:shape:0=api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Uapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
capi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasteapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumiapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2iapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle5api_smiles_encoder/StringSplit/StringSplitV2:values:0Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&api_smiles_encoder/string_lookup/EqualEqual5api_smiles_encoder/StringSplit/StringSplitV2:values:0(api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/SelectV2SelectV2*api_smiles_encoder/string_lookup/Equal:z:0+api_smiles_encoder_string_lookup_selectv2_tGapi_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/IdentityIdentity2api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0api_smiles_encoder/RaggedToTensor/Const:output:02api_smiles_encoder/string_lookup/Identity:output:08api_smiles_encoder/RaggedToTensor/default_value:output:07api_smiles_encoder/StringSplit/strided_slice_1:output:05api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSe
$cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs_1-cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,cof_smiles_encoder/StringSplit/strided_sliceStridedSlice6cof_smiles_encoder/StringSplit/StringSplitV2:indices:0;cof_smiles_encoder/StringSplit/strided_slice/stack:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.cof_smiles_encoder/StringSplit/strided_slice_1StridedSlice4cof_smiles_encoder/StringSplit/StringSplitV2:shape:0=cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Ucof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ccof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumicof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2icof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5cof_smiles_encoder/StringSplit/StringSplitV2:values:0Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
(cof_smiles_encoder/string_lookup_1/EqualEqual5cof_smiles_encoder/StringSplit/StringSplitV2:values:0*cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/SelectV2SelectV2,cof_smiles_encoder/string_lookup_1/Equal:z:0-cof_smiles_encoder_string_lookup_1_selectv2_tIcof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/IdentityIdentity4cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0cof_smiles_encoder/RaggedToTensor/Const:output:04cof_smiles_encoder/string_lookup_1/Identity:output:08cof_smiles_encoder/RaggedToTensor/default_value:output:07cof_smiles_encoder/StringSplit/strided_slice_1:output:05cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
!embedding/StatefulPartitionedCallStatefulPartitionedCall?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_7007*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_7006V
embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding/NotEqualNotEqual?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1_7022*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������P *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_7021X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqual?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
!api_cnn_0/StatefulPartitionedCallStatefulPartitionedCall*embedding/StatefulPartitionedCall:output:0api_cnn_0_7044api_cnn_0_7046*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_7043�
!cof_cnn_0/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0cof_cnn_0_7066cof_cnn_0_7068*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_7065�
!api_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*api_cnn_0/StatefulPartitionedCall:output:0api_cnn_1_7088api_cnn_1_7090*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_7087�
!cof_cnn_1/StatefulPartitionedCallStatefulPartitionedCall*cof_cnn_0/StatefulPartitionedCall:output:0cof_cnn_1_7110cof_cnn_1_7112*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������J�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_7109�
$global_max_pooling1d/PartitionedCallPartitionedCall*api_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_7120�
&global_max_pooling1d_1/PartitionedCallPartitionedCall*cof_cnn_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_7127�
concatenate/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0/global_max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_concatenate_layer_call_and_return_conditional_losses_7136�
+interaction_dense_0/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0interaction_dense_0_7150interaction_dense_0_7152*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_7149�
dropout/PartitionedCallPartitionedCall4interaction_dense_0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_7160�
+interaction_dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0interaction_dense_1_7174interaction_dense_1_7176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_7173�
dropout_1/PartitionedCallPartitionedCall4interaction_dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7184�
+interaction_dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0interaction_dense_2_7198interaction_dense_2_7200*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_7197�
dropout_2/PartitionedCallPartitionedCall4interaction_dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7208�
'prediction_head/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0prediction_head_7222prediction_head_7224*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_prediction_head_layer_call_and_return_conditional_losses_7221
IdentityIdentity0prediction_head/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^api_cnn_0/StatefulPartitionedCall"^api_cnn_1/StatefulPartitionedCall?^api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2"^cof_cnn_0/StatefulPartitionedCall"^cof_cnn_1/StatefulPartitionedCallA^cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall,^interaction_dense_0/StatefulPartitionedCall,^interaction_dense_1/StatefulPartitionedCall,^interaction_dense_2/StatefulPartitionedCall(^prediction_head/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2F
!api_cnn_0/StatefulPartitionedCall!api_cnn_0/StatefulPartitionedCall2F
!api_cnn_1/StatefulPartitionedCall!api_cnn_1/StatefulPartitionedCall2�
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22F
!cof_cnn_0/StatefulPartitionedCall!cof_cnn_0/StatefulPartitionedCall2F
!cof_cnn_1/StatefulPartitionedCall!cof_cnn_1/StatefulPartitionedCall2�
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV22F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2Z
+interaction_dense_0/StatefulPartitionedCall+interaction_dense_0/StatefulPartitionedCall2Z
+interaction_dense_1/StatefulPartitionedCall+interaction_dense_1/StatefulPartitionedCall2Z
+interaction_dense_2/StatefulPartitionedCall+interaction_dense_2/StatefulPartitionedCall2R
'prediction_head/StatefulPartitionedCall'prediction_head/StatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs:KG
#
_output_shapes
:���������
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

�
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_7197

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_8731

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
9
__inference__creator_9038
identity��
hash_table�

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*9
shared_name*(hash_table_./data/vocabulary.txt_-2_-1_2*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
�

�
I__inference_prediction_head_layer_call_and_return_conditional_losses_8792

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_deep_cocrystal_layer_call_fn_8274
inputs_0
inputs_1
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7:! 
	unknown_8:!  
	unknown_9: �

unknown_10:	�!

unknown_11: �

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*'
Tin 
2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:���������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
�
"__inference_signature_wrapper_8158
input_1
input_2
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7:! 
	unknown_8:!  
	unknown_9: �

unknown_10:	�!

unknown_11: �

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*'
Tin 
2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__wrapped_model_6872o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:LH
#
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_6882

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
o
E__inference_concatenate_layer_call_and_return_conditional_losses_7136

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_8842

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������J�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������J��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������M�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:���������M�
 
_user_specified_nameinputs
�
�/
 __inference__traced_restore_9513
file_prefixF
4assignvariableop_deep_cocrystal_embedding_embeddings:! J
8assignvariableop_1_deep_cocrystal_embedding_1_embeddings:! K
8assignvariableop_2_deep_cocrystal_prediction_head_kernel:	�D
6assignvariableop_3_deep_cocrystal_prediction_head_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: I
2assignvariableop_9_deep_cocrystal_api_cnn_0_kernel: �@
1assignvariableop_10_deep_cocrystal_api_cnn_0_bias:	�K
3assignvariableop_11_deep_cocrystal_api_cnn_1_kernel:��@
1assignvariableop_12_deep_cocrystal_api_cnn_1_bias:	�J
3assignvariableop_13_deep_cocrystal_cof_cnn_0_kernel: �@
1assignvariableop_14_deep_cocrystal_cof_cnn_0_bias:	�K
3assignvariableop_15_deep_cocrystal_cof_cnn_1_kernel:��@
1assignvariableop_16_deep_cocrystal_cof_cnn_1_bias:	�Q
=assignvariableop_17_deep_cocrystal_interaction_dense_0_kernel:
��J
;assignvariableop_18_deep_cocrystal_interaction_dense_0_bias:	�Q
=assignvariableop_19_deep_cocrystal_interaction_dense_1_kernel:
��J
;assignvariableop_20_deep_cocrystal_interaction_dense_1_bias:	�Q
=assignvariableop_21_deep_cocrystal_interaction_dense_2_kernel:
��J
;assignvariableop_22_deep_cocrystal_interaction_dense_2_bias:	�#
assignvariableop_23_total: #
assignvariableop_24_count: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: P
>assignvariableop_27_adam_deep_cocrystal_embedding_embeddings_m:! R
@assignvariableop_28_adam_deep_cocrystal_embedding_1_embeddings_m:! S
@assignvariableop_29_adam_deep_cocrystal_prediction_head_kernel_m:	�L
>assignvariableop_30_adam_deep_cocrystal_prediction_head_bias_m:Q
:assignvariableop_31_adam_deep_cocrystal_api_cnn_0_kernel_m: �G
8assignvariableop_32_adam_deep_cocrystal_api_cnn_0_bias_m:	�R
:assignvariableop_33_adam_deep_cocrystal_api_cnn_1_kernel_m:��G
8assignvariableop_34_adam_deep_cocrystal_api_cnn_1_bias_m:	�Q
:assignvariableop_35_adam_deep_cocrystal_cof_cnn_0_kernel_m: �G
8assignvariableop_36_adam_deep_cocrystal_cof_cnn_0_bias_m:	�R
:assignvariableop_37_adam_deep_cocrystal_cof_cnn_1_kernel_m:��G
8assignvariableop_38_adam_deep_cocrystal_cof_cnn_1_bias_m:	�X
Dassignvariableop_39_adam_deep_cocrystal_interaction_dense_0_kernel_m:
��Q
Bassignvariableop_40_adam_deep_cocrystal_interaction_dense_0_bias_m:	�X
Dassignvariableop_41_adam_deep_cocrystal_interaction_dense_1_kernel_m:
��Q
Bassignvariableop_42_adam_deep_cocrystal_interaction_dense_1_bias_m:	�X
Dassignvariableop_43_adam_deep_cocrystal_interaction_dense_2_kernel_m:
��Q
Bassignvariableop_44_adam_deep_cocrystal_interaction_dense_2_bias_m:	�P
>assignvariableop_45_adam_deep_cocrystal_embedding_embeddings_v:! R
@assignvariableop_46_adam_deep_cocrystal_embedding_1_embeddings_v:! S
@assignvariableop_47_adam_deep_cocrystal_prediction_head_kernel_v:	�L
>assignvariableop_48_adam_deep_cocrystal_prediction_head_bias_v:Q
:assignvariableop_49_adam_deep_cocrystal_api_cnn_0_kernel_v: �G
8assignvariableop_50_adam_deep_cocrystal_api_cnn_0_bias_v:	�R
:assignvariableop_51_adam_deep_cocrystal_api_cnn_1_kernel_v:��G
8assignvariableop_52_adam_deep_cocrystal_api_cnn_1_bias_v:	�Q
:assignvariableop_53_adam_deep_cocrystal_cof_cnn_0_kernel_v: �G
8assignvariableop_54_adam_deep_cocrystal_cof_cnn_0_bias_v:	�R
:assignvariableop_55_adam_deep_cocrystal_cof_cnn_1_kernel_v:��G
8assignvariableop_56_adam_deep_cocrystal_cof_cnn_1_bias_v:	�X
Dassignvariableop_57_adam_deep_cocrystal_interaction_dense_0_kernel_v:
��Q
Bassignvariableop_58_adam_deep_cocrystal_interaction_dense_0_bias_v:	�X
Dassignvariableop_59_adam_deep_cocrystal_interaction_dense_1_kernel_v:
��Q
Bassignvariableop_60_adam_deep_cocrystal_interaction_dense_1_bias_v:	�X
Dassignvariableop_61_adam_deep_cocrystal_interaction_dense_2_kernel_v:
��Q
Bassignvariableop_62_adam_deep_cocrystal_interaction_dense_2_bias_v:	�
identity_64��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B3api_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB8coformer_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB1prediction_head/kernel/.ATTRIBUTES/VARIABLE_VALUEB/prediction_head/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBOapi_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBTcoformer_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMprediction_head/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKprediction_head/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBOapi_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBTcoformer_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMprediction_head/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKprediction_head/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*�
value�B�@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp4assignvariableop_deep_cocrystal_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp8assignvariableop_1_deep_cocrystal_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp8assignvariableop_2_deep_cocrystal_prediction_head_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp6assignvariableop_3_deep_cocrystal_prediction_head_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp2assignvariableop_9_deep_cocrystal_api_cnn_0_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp1assignvariableop_10_deep_cocrystal_api_cnn_0_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp3assignvariableop_11_deep_cocrystal_api_cnn_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp1assignvariableop_12_deep_cocrystal_api_cnn_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp3assignvariableop_13_deep_cocrystal_cof_cnn_0_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_deep_cocrystal_cof_cnn_0_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp3assignvariableop_15_deep_cocrystal_cof_cnn_1_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_deep_cocrystal_cof_cnn_1_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp=assignvariableop_17_deep_cocrystal_interaction_dense_0_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp;assignvariableop_18_deep_cocrystal_interaction_dense_0_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp=assignvariableop_19_deep_cocrystal_interaction_dense_1_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp;assignvariableop_20_deep_cocrystal_interaction_dense_1_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp=assignvariableop_21_deep_cocrystal_interaction_dense_2_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp;assignvariableop_22_deep_cocrystal_interaction_dense_2_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_deep_cocrystal_embedding_embeddings_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_deep_cocrystal_embedding_1_embeddings_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp@assignvariableop_29_adam_deep_cocrystal_prediction_head_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_deep_cocrystal_prediction_head_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp:assignvariableop_31_adam_deep_cocrystal_api_cnn_0_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp8assignvariableop_32_adam_deep_cocrystal_api_cnn_0_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp:assignvariableop_33_adam_deep_cocrystal_api_cnn_1_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp8assignvariableop_34_adam_deep_cocrystal_api_cnn_1_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp:assignvariableop_35_adam_deep_cocrystal_cof_cnn_0_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp8assignvariableop_36_adam_deep_cocrystal_cof_cnn_0_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp:assignvariableop_37_adam_deep_cocrystal_cof_cnn_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp8assignvariableop_38_adam_deep_cocrystal_cof_cnn_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpDassignvariableop_39_adam_deep_cocrystal_interaction_dense_0_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOpBassignvariableop_40_adam_deep_cocrystal_interaction_dense_0_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOpDassignvariableop_41_adam_deep_cocrystal_interaction_dense_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpBassignvariableop_42_adam_deep_cocrystal_interaction_dense_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpDassignvariableop_43_adam_deep_cocrystal_interaction_dense_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpBassignvariableop_44_adam_deep_cocrystal_interaction_dense_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp>assignvariableop_45_adam_deep_cocrystal_embedding_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_deep_cocrystal_embedding_1_embeddings_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp@assignvariableop_47_adam_deep_cocrystal_prediction_head_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_deep_cocrystal_prediction_head_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp:assignvariableop_49_adam_deep_cocrystal_api_cnn_0_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp8assignvariableop_50_adam_deep_cocrystal_api_cnn_0_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp:assignvariableop_51_adam_deep_cocrystal_api_cnn_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp8assignvariableop_52_adam_deep_cocrystal_api_cnn_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp:assignvariableop_53_adam_deep_cocrystal_cof_cnn_0_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp8assignvariableop_54_adam_deep_cocrystal_cof_cnn_0_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp:assignvariableop_55_adam_deep_cocrystal_cof_cnn_1_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp8assignvariableop_56_adam_deep_cocrystal_cof_cnn_1_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpDassignvariableop_57_adam_deep_cocrystal_interaction_dense_0_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpBassignvariableop_58_adam_deep_cocrystal_interaction_dense_0_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpDassignvariableop_59_adam_deep_cocrystal_interaction_dense_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpBassignvariableop_60_adam_deep_cocrystal_interaction_dense_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpDassignvariableop_61_adam_deep_cocrystal_interaction_dense_2_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpBassignvariableop_62_adam_deep_cocrystal_interaction_dense_2_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_64Identity_64:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_cof_cnn_0_layer_call_fn_8851

inputs
unknown: �
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_7065t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������M�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������P 
 
_user_specified_nameinputs
�
�
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_8892

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������J�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������J��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������M�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:���������M�
 
_user_specified_nameinputs
�
�
2__inference_interaction_dense_1_layer_call_fn_8948

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_7173p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
D
(__inference_dropout_2_layer_call_fn_9011

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_7208a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
j
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_8737

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:����������U
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:���������J�:T P
,
_output_shapes
:���������J�
 
_user_specified_nameinputs
�
�
2__inference_interaction_dense_2_layer_call_fn_8995

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_7197p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
-__inference_deep_cocrystal_layer_call_fn_7790
input_1
input_2
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7:! 
	unknown_8:!  
	unknown_9: �

unknown_10:	�!

unknown_11: �

unknown_12:	�"

unknown_13:��

unknown_14:	�"

unknown_15:��

unknown_16:	�

unknown_17:
��

unknown_18:	�

unknown_19:
��

unknown_20:	�

unknown_21:
��

unknown_22:	�

unknown_23:	�

unknown_24:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*'
Tin 
2				*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*4
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7677o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
#
_output_shapes
:���������
!
_user_specified_name	input_1:LH
#
_output_shapes
:���������
!
_user_specified_name	input_2:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�

b
C__inference_dropout_2_layer_call_and_return_conditional_losses_7313

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:�
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:����������j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:����������Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
O
3__inference_global_max_pooling1d_layer_call_fn_8720

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *W
fRRP
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_6882i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
a
(__inference_dropout_1_layer_call_fn_8969

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_7346p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
__inference_<lambda>_9081!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identity��,text_file_init/InitializeTableFromTextFileV2�
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_index���������*
offset*
value_index���������J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
��
�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8683
inputs_0
inputs_1O
Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handleP
Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value	,
(api_smiles_encoder_string_lookup_equal_y/
+api_smiles_encoder_string_lookup_selectv2_t	Q
Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handleR
Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value	.
*cof_smiles_encoder_string_lookup_1_equal_y1
-cof_smiles_encoder_string_lookup_1_selectv2_t	1
embedding_embedding_lookup_8562:! 3
!embedding_1_embedding_lookup_8569:! L
5api_cnn_0_conv1d_expanddims_1_readvariableop_resource: �8
)api_cnn_0_biasadd_readvariableop_resource:	�L
5cof_cnn_0_conv1d_expanddims_1_readvariableop_resource: �8
)cof_cnn_0_biasadd_readvariableop_resource:	�M
5api_cnn_1_conv1d_expanddims_1_readvariableop_resource:��8
)api_cnn_1_biasadd_readvariableop_resource:	�M
5cof_cnn_1_conv1d_expanddims_1_readvariableop_resource:��8
)cof_cnn_1_biasadd_readvariableop_resource:	�F
2interaction_dense_0_matmul_readvariableop_resource:
��B
3interaction_dense_0_biasadd_readvariableop_resource:	�F
2interaction_dense_1_matmul_readvariableop_resource:
��B
3interaction_dense_1_biasadd_readvariableop_resource:	�F
2interaction_dense_2_matmul_readvariableop_resource:
��B
3interaction_dense_2_biasadd_readvariableop_resource:	�A
.prediction_head_matmul_readvariableop_resource:	�=
/prediction_head_biasadd_readvariableop_resource:
identity�� api_cnn_0/BiasAdd/ReadVariableOp�,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp� api_cnn_1/BiasAdd/ReadVariableOp�,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp�>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2� cof_cnn_0/BiasAdd/ReadVariableOp�,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp� cof_cnn_1/BiasAdd/ReadVariableOp�,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp�@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2�embedding/embedding_lookup�embedding_1/embedding_lookup�*interaction_dense_0/BiasAdd/ReadVariableOp�)interaction_dense_0/MatMul/ReadVariableOp�*interaction_dense_1/BiasAdd/ReadVariableOp�)interaction_dense_1/MatMul/ReadVariableOp�*interaction_dense_2/BiasAdd/ReadVariableOp�)interaction_dense_2/MatMul/ReadVariableOp�&prediction_head/BiasAdd/ReadVariableOp�%prediction_head/MatMul/ReadVariableOpe
$api_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,api_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs_0-api_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2api_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4api_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4api_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,api_smiles_encoder/StringSplit/strided_sliceStridedSlice6api_smiles_encoder/StringSplit/StringSplitV2:indices:0;api_smiles_encoder/StringSplit/strided_slice/stack:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=api_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4api_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6api_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6api_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.api_smiles_encoder/StringSplit/strided_slice_1StridedSlice4api_smiles_encoder/StringSplit/StringSplitV2:shape:0=api_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?api_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Uapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5api_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7api_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
capi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCasteapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
aapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0japi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumiapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2iapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]api_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0eapi_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2LookupTableFindV2Kapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_table_handle5api_smiles_encoder/StringSplit/StringSplitV2:values:0Lapi_smiles_encoder_string_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
&api_smiles_encoder/string_lookup/EqualEqual5api_smiles_encoder/StringSplit/StringSplitV2:values:0(api_smiles_encoder_string_lookup_equal_y*
T0*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/SelectV2SelectV2*api_smiles_encoder/string_lookup/Equal:z:0+api_smiles_encoder_string_lookup_selectv2_tGapi_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
)api_smiles_encoder/string_lookup/IdentityIdentity2api_smiles_encoder/string_lookup/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/api_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'api_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6api_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0api_smiles_encoder/RaggedToTensor/Const:output:02api_smiles_encoder/string_lookup/Identity:output:08api_smiles_encoder/RaggedToTensor/default_value:output:07api_smiles_encoder/StringSplit/strided_slice_1:output:05api_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDSe
$cof_smiles_encoder/StringSplit/ConstConst*
_output_shapes
: *
dtype0*
valueB B �
,cof_smiles_encoder/StringSplit/StringSplitV2StringSplitV2inputs_1-cof_smiles_encoder/StringSplit/Const:output:0*<
_output_shapes*
(:���������:���������:�
2cof_smiles_encoder/StringSplit/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        �
4cof_smiles_encoder/StringSplit/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       �
4cof_smiles_encoder/StringSplit/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
,cof_smiles_encoder/StringSplit/strided_sliceStridedSlice6cof_smiles_encoder/StringSplit/StringSplitV2:indices:0;cof_smiles_encoder/StringSplit/strided_slice/stack:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_1:output:0=cof_smiles_encoder/StringSplit/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask~
4cof_smiles_encoder/StringSplit/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:�
6cof_smiles_encoder/StringSplit/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
.cof_smiles_encoder/StringSplit/strided_slice_1StridedSlice4cof_smiles_encoder/StringSplit/StringSplitV2:shape:0=cof_smiles_encoder/StringSplit/strided_slice_1/stack:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_1:output:0?cof_smiles_encoder/StringSplit/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
: *
shrink_axis_mask�
Ucof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CastCast5cof_smiles_encoder/StringSplit/strided_slice:output:0*

DstT0*

SrcT0	*#
_output_shapes
:����������
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1Cast7cof_smiles_encoder/StringSplit/strided_slice_1:output:0*

DstT0*

SrcT0	*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ShapeShapeYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0*
T0*
_output_shapes
:�
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/ProdProdhcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Shape:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const:output:0*
T0*
_output_shapes
: �
ccof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/GreaterGreatergcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Prod:output:0lcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater/y:output:0*
T0*
_output_shapes
: �
^cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/CastCastecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Greater:z:0*

DstT0*

SrcT0
*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaxMaxYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_1:output:0*
T0*
_output_shapes
: �
_cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/addAddV2fcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Max:output:0hcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add/y:output:0*
T0*
_output_shapes
: �
]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mulMulbcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Cast:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/add:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MaximumMaximum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/mul:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/MinimumMinimum[cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast_1:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Maximum:z:0*
T0*
_output_shapes
: �
acof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2Const*
_output_shapes
: *
dtype0	*
valueB	 �
bcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/BincountBincountYcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cast:y:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Minimum:z:0jcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Const_2:output:0*
T0	*#
_output_shapes
:����������
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/CumsumCumsumicof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/bincount/Bincount:bins:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum/axis:output:0*
T0	*#
_output_shapes
:����������
`cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0Const*
_output_shapes
:*
dtype0	*
valueB	R �
\cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Wcof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concatConcatV2icof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/values_0:output:0]cof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/Cumsum:out:0ecof_smiles_encoder/StringSplit/RaggedFromValueRowIds/RowPartitionFromValueRowIds/concat/axis:output:0*
N*
T0	*#
_output_shapes
:����������
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2LookupTableFindV2Mcof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_table_handle5cof_smiles_encoder/StringSplit/StringSplitV2:values:0Ncof_smiles_encoder_string_lookup_1_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*#
_output_shapes
:����������
(cof_smiles_encoder/string_lookup_1/EqualEqual5cof_smiles_encoder/StringSplit/StringSplitV2:values:0*cof_smiles_encoder_string_lookup_1_equal_y*
T0*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/SelectV2SelectV2,cof_smiles_encoder/string_lookup_1/Equal:z:0-cof_smiles_encoder_string_lookup_1_selectv2_tIcof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:����������
+cof_smiles_encoder/string_lookup_1/IdentityIdentity4cof_smiles_encoder/string_lookup_1/SelectV2:output:0*
T0	*#
_output_shapes
:���������q
/cof_smiles_encoder/RaggedToTensor/default_valueConst*
_output_shapes
: *
dtype0	*
value	B	 R �
'cof_smiles_encoder/RaggedToTensor/ConstConst*
_output_shapes
:*
dtype0	*%
valueB	"��������P       �
6cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensorRaggedTensorToTensor0cof_smiles_encoder/RaggedToTensor/Const:output:04cof_smiles_encoder/string_lookup_1/Identity:output:08cof_smiles_encoder/RaggedToTensor/default_value:output:07cof_smiles_encoder/StringSplit/strided_slice_1:output:05cof_smiles_encoder/StringSplit/strided_slice:output:0*
T0	*
Tindex0	*
Tshape0	*'
_output_shapes
:���������P*
num_row_partition_tensors*7
row_partition_types 
FIRST_DIM_SIZEVALUE_ROWIDS�
embedding/embedding_lookupResourceGatherembedding_embedding_lookup_8562?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*2
_class(
&$loc:@embedding/embedding_lookup/8562*+
_output_shapes
:���������P *
dtype0�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*2
_class(
&$loc:@embedding/embedding_lookup/8562*+
_output_shapes
:���������P �
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P V
embedding/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding/NotEqualNotEqual?api_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������P�
embedding_1/embedding_lookupResourceGather!embedding_1_embedding_lookup_8569?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0*
Tindices0	*4
_class*
(&loc:@embedding_1/embedding_lookup/8569*+
_output_shapes
:���������P *
dtype0�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*4
_class*
(&loc:@embedding_1/embedding_lookup/8569*+
_output_shapes
:���������P �
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:���������P X
embedding_1/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R �
embedding_1/NotEqualNotEqual?cof_smiles_encoder/RaggedToTensor/RaggedTensorToTensor:result:0embedding_1/NotEqual/y:output:0*
T0	*'
_output_shapes
:���������Pj
api_cnn_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
api_cnn_0/Conv1D/ExpandDims
ExpandDims.embedding/embedding_lookup/Identity_1:output:0(api_cnn_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5api_cnn_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0c
!api_cnn_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
api_cnn_0/Conv1D/ExpandDims_1
ExpandDims4api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0*api_cnn_0/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
api_cnn_0/Conv1DConv2D$api_cnn_0/Conv1D/ExpandDims:output:0&api_cnn_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
api_cnn_0/Conv1D/SqueezeSqueezeapi_cnn_0/Conv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

����������
 api_cnn_0/BiasAdd/ReadVariableOpReadVariableOp)api_cnn_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
api_cnn_0/BiasAddBiasAdd!api_cnn_0/Conv1D/Squeeze:output:0(api_cnn_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�i
api_cnn_0/SeluSeluapi_cnn_0/BiasAdd:output:0*
T0*,
_output_shapes
:���������M�j
cof_cnn_0/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
cof_cnn_0/Conv1D/ExpandDims
ExpandDims0embedding_1/embedding_lookup/Identity_1:output:0(cof_cnn_0/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:���������P �
,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5cof_cnn_0_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
: �*
dtype0c
!cof_cnn_0/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
cof_cnn_0/Conv1D/ExpandDims_1
ExpandDims4cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp:value:0*cof_cnn_0/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
: ��
cof_cnn_0/Conv1DConv2D$cof_cnn_0/Conv1D/ExpandDims:output:0&cof_cnn_0/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������M�*
paddingVALID*
strides
�
cof_cnn_0/Conv1D/SqueezeSqueezecof_cnn_0/Conv1D:output:0*
T0*,
_output_shapes
:���������M�*
squeeze_dims

����������
 cof_cnn_0/BiasAdd/ReadVariableOpReadVariableOp)cof_cnn_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cof_cnn_0/BiasAddBiasAdd!cof_cnn_0/Conv1D/Squeeze:output:0(cof_cnn_0/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������M�i
cof_cnn_0/SeluSelucof_cnn_0/BiasAdd:output:0*
T0*,
_output_shapes
:���������M�j
api_cnn_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
api_cnn_1/Conv1D/ExpandDims
ExpandDimsapi_cnn_0/Selu:activations:0(api_cnn_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5api_cnn_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0c
!api_cnn_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
api_cnn_1/Conv1D/ExpandDims_1
ExpandDims4api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*api_cnn_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
api_cnn_1/Conv1DConv2D$api_cnn_1/Conv1D/ExpandDims:output:0&api_cnn_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
api_cnn_1/Conv1D/SqueezeSqueezeapi_cnn_1/Conv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

����������
 api_cnn_1/BiasAdd/ReadVariableOpReadVariableOp)api_cnn_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
api_cnn_1/BiasAddBiasAdd!api_cnn_1/Conv1D/Squeeze:output:0(api_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�i
api_cnn_1/SeluSeluapi_cnn_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������J�j
cof_cnn_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
cof_cnn_1/Conv1D/ExpandDims
ExpandDimscof_cnn_0/Selu:activations:0(cof_cnn_1/Conv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp5cof_cnn_1_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0c
!cof_cnn_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
cof_cnn_1/Conv1D/ExpandDims_1
ExpandDims4cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0*cof_cnn_1/Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
cof_cnn_1/Conv1DConv2D$cof_cnn_1/Conv1D/ExpandDims:output:0&cof_cnn_1/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
cof_cnn_1/Conv1D/SqueezeSqueezecof_cnn_1/Conv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

����������
 cof_cnn_1/BiasAdd/ReadVariableOpReadVariableOp)cof_cnn_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cof_cnn_1/BiasAddBiasAdd!cof_cnn_1/Conv1D/Squeeze:output:0(cof_cnn_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�i
cof_cnn_1/SeluSelucof_cnn_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������J�l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_max_pooling1d/MaxMaxapi_cnn_1/Selu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������n
,global_max_pooling1d_1/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
global_max_pooling1d_1/MaxMaxcof_cnn_1/Selu:activations:05global_max_pooling1d_1/Max/reduction_indices:output:0*
T0*(
_output_shapes
:����������Y
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatenate/concatConcatV2!global_max_pooling1d/Max:output:0#global_max_pooling1d_1/Max:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
)interaction_dense_0/MatMul/ReadVariableOpReadVariableOp2interaction_dense_0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
interaction_dense_0/MatMulMatMulconcatenate/concat:output:01interaction_dense_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*interaction_dense_0/BiasAdd/ReadVariableOpReadVariableOp3interaction_dense_0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
interaction_dense_0/BiasAddBiasAdd$interaction_dense_0/MatMul:product:02interaction_dense_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
interaction_dense_0/ReluRelu$interaction_dense_0/BiasAdd:output:0*
T0*(
_output_shapes
:����������Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout/dropout/MulMul&interaction_dense_0/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:����������k
dropout/dropout/ShapeShape&interaction_dense_0/Relu:activations:0*
T0*
_output_shapes
:�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed*c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
)interaction_dense_1/MatMul/ReadVariableOpReadVariableOp2interaction_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
interaction_dense_1/MatMulMatMuldropout/dropout/Mul_1:z:01interaction_dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*interaction_dense_1/BiasAdd/ReadVariableOpReadVariableOp3interaction_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
interaction_dense_1/BiasAddBiasAdd$interaction_dense_1/MatMul:product:02interaction_dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
interaction_dense_1/ReluRelu$interaction_dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_1/dropout/MulMul&interaction_dense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:����������m
dropout_1/dropout/ShapeShape&interaction_dense_1/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed**
seed2e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
)interaction_dense_2/MatMul/ReadVariableOpReadVariableOp2interaction_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
interaction_dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:01interaction_dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
*interaction_dense_2/BiasAdd/ReadVariableOpReadVariableOp3interaction_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
interaction_dense_2/BiasAddBiasAdd$interaction_dense_2/MatMul:product:02interaction_dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������y
interaction_dense_2/ReluRelu$interaction_dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:����������\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?�
dropout_2/dropout/MulMul&interaction_dense_2/Relu:activations:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:����������m
dropout_2/dropout/ShapeShape&interaction_dense_2/Relu:activations:0*
T0*
_output_shapes
:�
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0*

seed**
seed2e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=�
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:�����������
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:�����������
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:�����������
%prediction_head/MatMul/ReadVariableOpReadVariableOp.prediction_head_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
prediction_head/MatMulMatMuldropout_2/dropout/Mul_1:z:0-prediction_head/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&prediction_head/BiasAdd/ReadVariableOpReadVariableOp/prediction_head_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
prediction_head/BiasAddBiasAdd prediction_head/MatMul:product:0.prediction_head/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
prediction_head/SigmoidSigmoid prediction_head/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentityprediction_head/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^api_cnn_0/BiasAdd/ReadVariableOp-^api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp!^api_cnn_1/BiasAdd/ReadVariableOp-^api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp?^api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2!^cof_cnn_0/BiasAdd/ReadVariableOp-^cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp!^cof_cnn_1/BiasAdd/ReadVariableOp-^cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOpA^cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2^embedding/embedding_lookup^embedding_1/embedding_lookup+^interaction_dense_0/BiasAdd/ReadVariableOp*^interaction_dense_0/MatMul/ReadVariableOp+^interaction_dense_1/BiasAdd/ReadVariableOp*^interaction_dense_1/MatMul/ReadVariableOp+^interaction_dense_2/BiasAdd/ReadVariableOp*^interaction_dense_2/MatMul/ReadVariableOp'^prediction_head/BiasAdd/ReadVariableOp&^prediction_head/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R:���������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 api_cnn_0/BiasAdd/ReadVariableOp api_cnn_0/BiasAdd/ReadVariableOp2\
,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp,api_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp2D
 api_cnn_1/BiasAdd/ReadVariableOp api_cnn_1/BiasAdd/ReadVariableOp2\
,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp,api_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp2�
>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV2>api_smiles_encoder/string_lookup/None_Lookup/LookupTableFindV22D
 cof_cnn_0/BiasAdd/ReadVariableOp cof_cnn_0/BiasAdd/ReadVariableOp2\
,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp,cof_cnn_0/Conv1D/ExpandDims_1/ReadVariableOp2D
 cof_cnn_1/BiasAdd/ReadVariableOp cof_cnn_1/BiasAdd/ReadVariableOp2\
,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp,cof_cnn_1/Conv1D/ExpandDims_1/ReadVariableOp2�
@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV2@cof_smiles_encoder/string_lookup_1/None_Lookup/LookupTableFindV228
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2X
*interaction_dense_0/BiasAdd/ReadVariableOp*interaction_dense_0/BiasAdd/ReadVariableOp2V
)interaction_dense_0/MatMul/ReadVariableOp)interaction_dense_0/MatMul/ReadVariableOp2X
*interaction_dense_1/BiasAdd/ReadVariableOp*interaction_dense_1/BiasAdd/ReadVariableOp2V
)interaction_dense_1/MatMul/ReadVariableOp)interaction_dense_1/MatMul/ReadVariableOp2X
*interaction_dense_2/BiasAdd/ReadVariableOp*interaction_dense_2/BiasAdd/ReadVariableOp2V
)interaction_dense_2/MatMul/ReadVariableOp)interaction_dense_2/MatMul/ReadVariableOp2P
&prediction_head/BiasAdd/ReadVariableOp&prediction_head/BiasAdd/ReadVariableOp2N
%prediction_head/MatMul/ReadVariableOp%prediction_head/MatMul/ReadVariableOp:M I
#
_output_shapes
:���������
"
_user_specified_name
inputs/0:MI
#
_output_shapes
:���������
"
_user_specified_name
inputs/1:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: 
�
�
(__inference_api_cnn_0_layer_call_fn_8801

inputs
unknown: �
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������M�*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_7043t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:���������M�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������P 
 
_user_specified_nameinputs
�
�
__inference__initializer_9045!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identity��,text_file_init/InitializeTableFromTextFileV2�
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_index���������*
offset*
value_index���������G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
�
l
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_8753

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
l
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_6895

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:������������������]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'���������������������������:e a
=
_output_shapes+
):'���������������������������
 
_user_specified_nameinputs
�
�
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_7087

inputsC
+conv1d_expanddims_1_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*0
_output_shapes
:���������M��
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:��*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : �
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:���
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:���������J�*
paddingVALID*
strides
�
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:���������J�*
squeeze_dims

���������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������J�U
SeluSeluBiasAdd:output:0*
T0*,
_output_shapes
:���������J�f
IdentityIdentitySelu:activations:0^NoOp*
T0*,
_output_shapes
:���������J��
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������M�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:���������M�
 
_user_specified_nameinputs
�

�
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_9006

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_3:0StatefulPartitionedCall_48"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
7
input_1,
serving_default_input_1:0���������
7
input_2,
serving_default_input_2:0���������>
output_12
StatefulPartitionedCall_2:0���������tensorflow/serving/predict2,

asset_path_initializer:0vocabulary.txt2.

asset_path_initializer_1:0vocabulary.txt:��
�
api_smiles_encoder
cof_smiles_encoder
api_embedding
coformer_embedding
	api_convs
	cof_convs
api_pooling
cof_pooling
	interaction_concat

interaction_denses
prediction_head
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_model
;
_lookup_layer
	keras_api"
_tf_keras_layer
;
_lookup_layer
	keras_api"
_tf_keras_layer
�

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
.
 0
!1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
J
00
11
22
33
44
55"
trackable_list_wrapper
�

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
<iter

=beta_1

>beta_2
	?decay
@learning_ratem�m�6m�7m�Am�Bm�Cm�Dm�Em�Fm�Gm�Hm�Im�Jm�Km�Lm�Mm�Nm�v�v�6v�7v�Av�Bv�Cv�Dv�Ev�Fv�Gv�Hv�Iv�Jv�Kv�Lv�Mv�Nv�"
	optimizer
�
0
1
A2
B3
C4
D5
E6
F7
G8
H9
I10
J11
K12
L13
M14
N15
616
717"
trackable_list_wrapper
�
0
1
A2
B3
C4
D5
E6
F7
G8
H9
I10
J11
K12
L13
M14
N15
616
717"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
:
Tlookup_table
U	keras_api"
_tf_keras_layer
"
_generic_user_object
:
Vlookup_table
W	keras_api"
_tf_keras_layer
"
_generic_user_object
5:3! 2#deep_cocrystal/embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
7:5! 2%deep_cocrystal/embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Akernel
Bbias
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ckernel
Dbias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Ekernel
Fbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Gkernel
Hbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
$	variables
%trainable_variables
&regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
(	variables
)trainable_variables
*regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Ikernel
Jbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Kkernel
Lbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Mkernel
Nbias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
8:6	�2%deep_cocrystal/prediction_head/kernel
1:/2#deep_cocrystal/prediction_head/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
8	variables
9trainable_variables
:regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
6:4 �2deep_cocrystal/api_cnn_0/kernel
,:*�2deep_cocrystal/api_cnn_0/bias
7:5��2deep_cocrystal/api_cnn_1/kernel
,:*�2deep_cocrystal/api_cnn_1/bias
6:4 �2deep_cocrystal/cof_cnn_0/kernel
,:*�2deep_cocrystal/cof_cnn_0/bias
7:5��2deep_cocrystal/cof_cnn_1/kernel
,:*�2deep_cocrystal/cof_cnn_1/bias
=:;
��2)deep_cocrystal/interaction_dense_0/kernel
6:4�2'deep_cocrystal/interaction_dense_0/bias
=:;
��2)deep_cocrystal/interaction_dense_1/kernel
6:4�2'deep_cocrystal/interaction_dense_1/bias
=:;
��2)deep_cocrystal/interaction_dense_2/kernel
6:4�2'deep_cocrystal/interaction_dense_2/bias
 "
trackable_list_wrapper
�
0
1
2
3
 4
!5
"6
#7
8
9
	10
011
112
213
314
415
516
17"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
n
�_initializer
�_create_resource
�_initialize
�_destroy_resourceR jCustom.StaticHashTable
"
_generic_user_object
n
�_initializer
�_create_resource
�_initialize
�_destroy_resourceR jCustom.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
n	variables
otrainable_variables
pregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
.
�	_filename"
_generic_user_object
.
�	_filename"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
* 
*
::8! 2*Adam/deep_cocrystal/embedding/embeddings/m
<::! 2,Adam/deep_cocrystal/embedding_1/embeddings/m
=:;	�2,Adam/deep_cocrystal/prediction_head/kernel/m
6:42*Adam/deep_cocrystal/prediction_head/bias/m
;:9 �2&Adam/deep_cocrystal/api_cnn_0/kernel/m
1:/�2$Adam/deep_cocrystal/api_cnn_0/bias/m
<::��2&Adam/deep_cocrystal/api_cnn_1/kernel/m
1:/�2$Adam/deep_cocrystal/api_cnn_1/bias/m
;:9 �2&Adam/deep_cocrystal/cof_cnn_0/kernel/m
1:/�2$Adam/deep_cocrystal/cof_cnn_0/bias/m
<::��2&Adam/deep_cocrystal/cof_cnn_1/kernel/m
1:/�2$Adam/deep_cocrystal/cof_cnn_1/bias/m
B:@
��20Adam/deep_cocrystal/interaction_dense_0/kernel/m
;:9�2.Adam/deep_cocrystal/interaction_dense_0/bias/m
B:@
��20Adam/deep_cocrystal/interaction_dense_1/kernel/m
;:9�2.Adam/deep_cocrystal/interaction_dense_1/bias/m
B:@
��20Adam/deep_cocrystal/interaction_dense_2/kernel/m
;:9�2.Adam/deep_cocrystal/interaction_dense_2/bias/m
::8! 2*Adam/deep_cocrystal/embedding/embeddings/v
<::! 2,Adam/deep_cocrystal/embedding_1/embeddings/v
=:;	�2,Adam/deep_cocrystal/prediction_head/kernel/v
6:42*Adam/deep_cocrystal/prediction_head/bias/v
;:9 �2&Adam/deep_cocrystal/api_cnn_0/kernel/v
1:/�2$Adam/deep_cocrystal/api_cnn_0/bias/v
<::��2&Adam/deep_cocrystal/api_cnn_1/kernel/v
1:/�2$Adam/deep_cocrystal/api_cnn_1/bias/v
;:9 �2&Adam/deep_cocrystal/cof_cnn_0/kernel/v
1:/�2$Adam/deep_cocrystal/cof_cnn_0/bias/v
<::��2&Adam/deep_cocrystal/cof_cnn_1/kernel/v
1:/�2$Adam/deep_cocrystal/cof_cnn_1/bias/v
B:@
��20Adam/deep_cocrystal/interaction_dense_0/kernel/v
;:9�2.Adam/deep_cocrystal/interaction_dense_0/bias/v
B:@
��20Adam/deep_cocrystal/interaction_dense_1/kernel/v
;:9�2.Adam/deep_cocrystal/interaction_dense_1/bias/v
B:@
��20Adam/deep_cocrystal/interaction_dense_2/kernel/v
;:9�2.Adam/deep_cocrystal/interaction_dense_2/bias/v
�2�
-__inference_deep_cocrystal_layer_call_fn_7283
-__inference_deep_cocrystal_layer_call_fn_8216
-__inference_deep_cocrystal_layer_call_fn_8274
-__inference_deep_cocrystal_layer_call_fn_7790�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8468
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8683
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7941
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
__inference__wrapped_model_6872input_1input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_embedding_layer_call_fn_8690�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_embedding_layer_call_and_return_conditional_losses_8699�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_1_layer_call_fn_8706�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_1_layer_call_and_return_conditional_losses_8715�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_global_max_pooling1d_layer_call_fn_8720
3__inference_global_max_pooling1d_layer_call_fn_8725�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_8731
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_8737�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_global_max_pooling1d_1_layer_call_fn_8742
5__inference_global_max_pooling1d_1_layer_call_fn_8747�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_8753
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_8759�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_concatenate_layer_call_fn_8765�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_concatenate_layer_call_and_return_conditional_losses_8772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_prediction_head_layer_call_fn_8781�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_prediction_head_layer_call_and_return_conditional_losses_8792�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_8158input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_api_cnn_0_layer_call_fn_8801�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_8817�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_api_cnn_1_layer_call_fn_8826�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_8842�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_cof_cnn_0_layer_call_fn_8851�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_8867�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_cof_cnn_1_layer_call_fn_8876�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_8892�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_interaction_dense_0_layer_call_fn_8901�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_8912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dropout_layer_call_fn_8917
&__inference_dropout_layer_call_fn_8922�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_dropout_layer_call_and_return_conditional_losses_8927
A__inference_dropout_layer_call_and_return_conditional_losses_8939�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_interaction_dense_1_layer_call_fn_8948�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_8959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dropout_1_layer_call_fn_8964
(__inference_dropout_1_layer_call_fn_8969�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_dropout_1_layer_call_and_return_conditional_losses_8974
C__inference_dropout_1_layer_call_and_return_conditional_losses_8986�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_interaction_dense_2_layer_call_fn_8995�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_9006�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dropout_2_layer_call_fn_9011
(__inference_dropout_2_layer_call_fn_9016�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_dropout_2_layer_call_and_return_conditional_losses_9021
C__inference_dropout_2_layer_call_and_return_conditional_losses_9033�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
__inference__creator_9038�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__initializer_9045�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__destroyer_9050�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__creator_9055�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__initializer_9062�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference__destroyer_9067�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_55
__inference__creator_9038�

� 
� "� 5
__inference__creator_9055�

� 
� "� 7
__inference__destroyer_9050�

� 
� "� 7
__inference__destroyer_9067�

� 
� "� >
__inference__initializer_9045�T�

� 
� "� >
__inference__initializer_9062�V�

� 
� "� �
__inference__wrapped_model_6872� T���V���ABEFCDGHIJKLMN67P�M
F�C
A�>
�
input_1���������
�
input_2���������
� "3�0
.
output_1"�
output_1����������
C__inference_api_cnn_0_layer_call_and_return_conditional_losses_8817eAB3�0
)�&
$�!
inputs���������P 
� "*�'
 �
0���������M�
� �
(__inference_api_cnn_0_layer_call_fn_8801XAB3�0
)�&
$�!
inputs���������P 
� "����������M��
C__inference_api_cnn_1_layer_call_and_return_conditional_losses_8842fCD4�1
*�'
%�"
inputs���������M�
� "*�'
 �
0���������J�
� �
(__inference_api_cnn_1_layer_call_fn_8826YCD4�1
*�'
%�"
inputs���������M�
� "����������J��
C__inference_cof_cnn_0_layer_call_and_return_conditional_losses_8867eEF3�0
)�&
$�!
inputs���������P 
� "*�'
 �
0���������M�
� �
(__inference_cof_cnn_0_layer_call_fn_8851XEF3�0
)�&
$�!
inputs���������P 
� "����������M��
C__inference_cof_cnn_1_layer_call_and_return_conditional_losses_8892fGH4�1
*�'
%�"
inputs���������M�
� "*�'
 �
0���������J�
� �
(__inference_cof_cnn_1_layer_call_fn_8876YGH4�1
*�'
%�"
inputs���������M�
� "����������J��
E__inference_concatenate_layer_call_and_return_conditional_losses_8772�\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "&�#
�
0����������
� �
*__inference_concatenate_layer_call_fn_8765y\�Y
R�O
M�J
#� 
inputs/0����������
#� 
inputs/1����������
� "������������
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_7941� T���V���ABEFCDGHIJKLMN67`�]
F�C
A�>
�
input_1���������
�
input_2���������
�

trainingp "%�"
�
0���������
� �
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8092� T���V���ABEFCDGHIJKLMN67`�]
F�C
A�>
�
input_1���������
�
input_2���������
�

trainingp"%�"
�
0���������
� �
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8468� T���V���ABEFCDGHIJKLMN67b�_
H�E
C�@
�
inputs/0���������
�
inputs/1���������
�

trainingp "%�"
�
0���������
� �
H__inference_deep_cocrystal_layer_call_and_return_conditional_losses_8683� T���V���ABEFCDGHIJKLMN67b�_
H�E
C�@
�
inputs/0���������
�
inputs/1���������
�

trainingp"%�"
�
0���������
� �
-__inference_deep_cocrystal_layer_call_fn_7283� T���V���ABEFCDGHIJKLMN67`�]
F�C
A�>
�
input_1���������
�
input_2���������
�

trainingp "�����������
-__inference_deep_cocrystal_layer_call_fn_7790� T���V���ABEFCDGHIJKLMN67`�]
F�C
A�>
�
input_1���������
�
input_2���������
�

trainingp"�����������
-__inference_deep_cocrystal_layer_call_fn_8216� T���V���ABEFCDGHIJKLMN67b�_
H�E
C�@
�
inputs/0���������
�
inputs/1���������
�

trainingp "�����������
-__inference_deep_cocrystal_layer_call_fn_8274� T���V���ABEFCDGHIJKLMN67b�_
H�E
C�@
�
inputs/0���������
�
inputs/1���������
�

trainingp"�����������
C__inference_dropout_1_layer_call_and_return_conditional_losses_8974^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
C__inference_dropout_1_layer_call_and_return_conditional_losses_8986^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� }
(__inference_dropout_1_layer_call_fn_8964Q4�1
*�'
!�
inputs����������
p 
� "�����������}
(__inference_dropout_1_layer_call_fn_8969Q4�1
*�'
!�
inputs����������
p
� "������������
C__inference_dropout_2_layer_call_and_return_conditional_losses_9021^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
C__inference_dropout_2_layer_call_and_return_conditional_losses_9033^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� }
(__inference_dropout_2_layer_call_fn_9011Q4�1
*�'
!�
inputs����������
p 
� "�����������}
(__inference_dropout_2_layer_call_fn_9016Q4�1
*�'
!�
inputs����������
p
� "������������
A__inference_dropout_layer_call_and_return_conditional_losses_8927^4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
A__inference_dropout_layer_call_and_return_conditional_losses_8939^4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� {
&__inference_dropout_layer_call_fn_8917Q4�1
*�'
!�
inputs����������
p 
� "�����������{
&__inference_dropout_layer_call_fn_8922Q4�1
*�'
!�
inputs����������
p
� "������������
E__inference_embedding_1_layer_call_and_return_conditional_losses_8715_/�,
%�"
 �
inputs���������P	
� ")�&
�
0���������P 
� �
*__inference_embedding_1_layer_call_fn_8706R/�,
%�"
 �
inputs���������P	
� "����������P �
C__inference_embedding_layer_call_and_return_conditional_losses_8699_/�,
%�"
 �
inputs���������P	
� ")�&
�
0���������P 
� ~
(__inference_embedding_layer_call_fn_8690R/�,
%�"
 �
inputs���������P	
� "����������P �
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_8753wE�B
;�8
6�3
inputs'���������������������������
� ".�+
$�!
0������������������
� �
P__inference_global_max_pooling1d_1_layer_call_and_return_conditional_losses_8759^4�1
*�'
%�"
inputs���������J�
� "&�#
�
0����������
� �
5__inference_global_max_pooling1d_1_layer_call_fn_8742jE�B
;�8
6�3
inputs'���������������������������
� "!��������������������
5__inference_global_max_pooling1d_1_layer_call_fn_8747Q4�1
*�'
%�"
inputs���������J�
� "������������
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_8731wE�B
;�8
6�3
inputs'���������������������������
� ".�+
$�!
0������������������
� �
N__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_8737^4�1
*�'
%�"
inputs���������J�
� "&�#
�
0����������
� �
3__inference_global_max_pooling1d_layer_call_fn_8720jE�B
;�8
6�3
inputs'���������������������������
� "!��������������������
3__inference_global_max_pooling1d_layer_call_fn_8725Q4�1
*�'
%�"
inputs���������J�
� "������������
M__inference_interaction_dense_0_layer_call_and_return_conditional_losses_8912^IJ0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
2__inference_interaction_dense_0_layer_call_fn_8901QIJ0�-
&�#
!�
inputs����������
� "������������
M__inference_interaction_dense_1_layer_call_and_return_conditional_losses_8959^KL0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
2__inference_interaction_dense_1_layer_call_fn_8948QKL0�-
&�#
!�
inputs����������
� "������������
M__inference_interaction_dense_2_layer_call_and_return_conditional_losses_9006^MN0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
2__inference_interaction_dense_2_layer_call_fn_8995QMN0�-
&�#
!�
inputs����������
� "������������
I__inference_prediction_head_layer_call_and_return_conditional_losses_8792]670�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
.__inference_prediction_head_layer_call_fn_8781P670�-
&�#
!�
inputs����������
� "�����������
"__inference_signature_wrapper_8158� T���V���ABEFCDGHIJKLMN67a�^
� 
W�T
(
input_1�
input_1���������
(
input_2�
input_2���������"3�0
.
output_1"�
output_1���������