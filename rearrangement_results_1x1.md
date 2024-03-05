Starting imports
Imports completed in 11.913735151290894 seconds
LocalArgs(lora_path='models/vit_finegrained_5x40_procthor.pt', test_folder_path='/scratch/aneesh.chavan/8-room-v1/1/', rearranged_test_folder_path='/scratch/aneesh.chavan/8-room-v1/1/', device='cuda', sam_checkpoint_path='/scratch/aneesh.chavan/sam_vit_h_4b8939.pth', ram_pretrained_path='/scratch/aneesh.chavan/ram_swin_large_14m.pth', sampling_period=5, downsampling_rate=5, save_dir='/scratch/aneesh.chavan/results', start_file_index=2, last_file_index=250, rot_correction=0.0, look_around_range=0, save_individual_objects=False, down_sample_voxel_size=0.01, create_ext_mesh=False, save_point_clouds=False, fpfh_global_dist_factor=1.5, fpfh_local_dist_factor=0.4, fpfh_voxel_size=0.05, localise_times=1, loc_results_start_file_index=1, loc_results_last_file_index=250, loc_results_sampling_period=10)
Created save directory /scratch/aneesh.chavan/results
We have 3315 files

Begin Memory Initialization
/encoder/layer/0/crossattention/self/query is tied
/encoder/layer/0/crossattention/self/key is tied
/encoder/layer/0/crossattention/self/value is tied
/encoder/layer/0/crossattention/output/dense is tied
/encoder/layer/0/crossattention/output/LayerNorm is tied
/encoder/layer/0/intermediate/dense is tied
/encoder/layer/0/output/dense is tied
/encoder/layer/0/output/LayerNorm is tied
/encoder/layer/1/crossattention/self/query is tied
/encoder/layer/1/crossattention/self/key is tied
/encoder/layer/1/crossattention/self/value is tied
/encoder/layer/1/crossattention/output/dense is tied
/encoder/layer/1/crossattention/output/LayerNorm is tied
/encoder/layer/1/intermediate/dense is tied
/encoder/layer/1/output/dense is tied
/encoder/layer/1/output/LayerNorm is tied
--------------
/scratch/aneesh.chavan/ram_swin_large_14m.pth
--------------
load checkpoint from /scratch/aneesh.chavan/ram_swin_large_14m.pth
vit: swin_l
final text_encoder_type: bert-base-uncased
Model loaded from /home2/aneesh.chavan/.cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swinb_cogcoor.pth 
 => _IncompatibleKeys(missing_keys=[], unexpected_keys=['label_enc.weight'])
SAM loaded
Memory Init'ed


	Seeing image 2 currently
caption post ram:  armchair  .  bin  .  chair  .  container  .  stool  
Memory usage: 9.766 GB
Max GPU memory usage: 9.282 GB
	 ----------------
Downsampling at 0 frame voxel size as 0.01

	Seeing image 7 currently
caption post ram:  armchair  .  chair  
Memory usage: 9.766 GB
Max GPU memory usage: 9.366 GB
	 ----------------

	Seeing image 12 currently
caption post ram:  armchair  .  barrel  .  bin  .  chair  .  container  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 17 currently
caption post ram:  armchair  .  barrel  .  bin  .  lamp  .  chair  .  container  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 22 currently
caption post ram:  armchair  .  chair  .  stand  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 27 currently
caption post ram:  armchair  .  bin  .  chair  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 5 frame voxel size as 0.01

	Seeing image 32 currently
caption post ram:  armchair  .  chair  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 37 currently
caption post ram:   lamp  .  plaster bandage  .  chair  .  easel  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 42 currently
caption post ram:  armchair  .  plaster bandage  .  chair  .  stool  
Memory usage: 9.767 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 47 currently
caption post ram:  chair  .  table  .  stool  
Memory usage: 9.768 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 52 currently
caption post ram:  lamp  .  easel  .  pyramid  
Memory usage: 9.775 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 10 frame voxel size as 0.01

	Seeing image 57 currently
caption post ram:  armchair  .  lamp  .  plaster bandage  .  chair  .  stool  
Memory usage: 9.775 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 62 currently
caption post ram:  lamp  .  chair  .  stand  .  stool  
Memory usage: 9.775 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 67 currently
caption post ram:  chair  .  stool  
Memory usage: 9.775 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 72 currently
caption post ram:  armchair  .  chair  .  stool  
Memory usage: 9.775 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 77 currently
caption post ram:  lamp  .  chair  .  pole  .  stand  .  stool 
Memory usage: 9.777 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 15 frame voxel size as 0.01

	Seeing image 82 currently
caption post ram:  armchair  .  lamp  .  chair  .  pillow  .  stool  
Memory usage: 9.777 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 87 currently
caption post ram:  lamp  .  plaster bandage  .  lamp post  .  pole  .  stand 
Memory usage: 9.776 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 92 currently
caption post ram:  lamp  .  plaster bandage  .  chair  .  stool  
Memory usage: 9.777 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 97 currently
caption post ram:  lamp  .  chair  .  stool  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 102 currently
caption post ram:  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 20 frame voxel size as 0.01

	Seeing image 107 currently
caption post ram:  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 112 currently
caption post ram:  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 117 currently
caption post ram:  armchair  .  lamp  .  chair  .  couch  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 122 currently
caption post ram:  armchair  .  chair  .  couch  .  stool  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 127 currently
caption post ram:  armchair  .  beach chair  .  chair  .  daybed  .  couch  .  stool  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 25 frame voxel size as 0.01

	Seeing image 132 currently
caption post ram:   ledge  .  skateboarder  .  trick 
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 137 currently
caption post ram:  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 142 currently
caption post ram:  bed  .  pillow  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 147 currently
caption post ram:  bed  .  chair  .  couch  .  pillow  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 152 currently
caption post ram:  armchair  .  chair  .  couch  .  table  .  dinning table  .  stool  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 30 frame voxel size as 0.01

	Seeing image 157 currently
caption post ram:  armchair  .  couch  .  pillow  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 162 currently
caption post ram:  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 167 currently
caption post ram:   skateboarder  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 172 currently
caption post ram:  armchair  .  chair  .  couch  .  hassock  .  pillow  
Memory usage: 9.778 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 177 currently
caption post ram:  couch  .  pillow  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 35 frame voxel size as 0.01

	Seeing image 182 currently
caption post ram:  armchair  .  lamp  .  chair  .  daybed  .  couch  .  hassock  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 187 currently
caption post ram:  armchair  .  lamp  .  chair  .  couch  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 192 currently
caption post ram:  box  .  cardboard box  .  chair  .  table  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 197 currently
caption post ram:  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 202 currently
caption post ram:  box  .  cardboard box  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 40 frame voxel size as 0.01

	Seeing image 207 currently
caption post ram:  armchair  .  lamp  .  chair  .  couch  .  hassock  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 212 currently
caption post ram:  armchair  .  chair  .  couch  .  hassock  .  pillow  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 217 currently
caption post ram:  armchair  .  bin  .  chair  .  couch  .  hassock  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 222 currently
caption post ram:  lamp  .  chair  .  table  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 227 currently
caption post ram:  bin  .  box  .  chair  .  container  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling at 45 frame voxel size as 0.01

	Seeing image 232 currently
caption post ram:  chair  .  table  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 237 currently
caption post ram:  bed  .  cabinet  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 242 currently
caption post ram:  bin  .  chair  .  container  .  recycling bin  .  stool  
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------

	Seeing image 247 currently
caption post ram:  armchair  .  bed  .  lamp  .  chair  .  couch  .  futon  .  stool 
Memory usage: 9.779 GB
Max GPU memory usage: 9.367 GB
	 ----------------
Downsampling using voxel size as 0.01
Traversal completed in 115.08949184417725 seconds
Memory's pointcloud saved to /scratch/aneesh.chavan/results/mem_2_250_5_0.pcd


	---------------------
Objects stored in memory:
ID: 0 | Names: [armchair ,lamp ,chair] |  Num embs: 18 | Pcd size: (3, 8750)
ID: 1 | Names: [container ,barrel ,armchair] |  Num embs: 5 | Pcd size: (3, 8038)
ID: 2 | Names: [plaster bandage] |  Num embs: 1 | Pcd size: (3, 3692)
ID: 3 | Names: [easel] |  Num embs: 1 | Pcd size: (3, 8351)
ID: 4 | Names: [pyramid] |  Num embs: 1 | Pcd size: (3, 2105)
ID: 5 | Names: [lamp] |  Num embs: 1 | Pcd size: (3, 3392)
ID: 6 | Names: [plaster bandage ,chair] |  Num embs: 2 | Pcd size: (3, 35030)
ID: 7 | Names: [armchair ,couch] |  Num embs: 8 | Pcd size: (3, 32063)
ID: 8 | Names: [lamp ,chair ,couch ,armchair ,beach chair ,bed] |  Num embs: 16 | Pcd size: (3, 34958)
ID: 9 | Names: [stool] |  Num embs: 2 | Pcd size: (3, 5968)
ID: 10 | Names: [stool ,bed] |  Num embs: 3 | Pcd size: (3, 4412)
ID: 11 | Names: [skateboarder] |  Num embs: 2 | Pcd size: (3, 4519)
ID: 12 | Names: [bed ,table ,armchair] |  Num embs: 4 | Pcd size: (3, 32026)
ID: 13 | Names: [pillow ,chair ,armchair] |  Num embs: 3 | Pcd size: (3, 11518)
ID: 14 | Names: [chair] |  Num embs: 1 | Pcd size: (3, 3756)
ID: 15 | Names: [table] |  Num embs: 1 | Pcd size: (3, 13143)
ID: 16 | Names: [armchair] |  Num embs: 2 | Pcd size: (3, 2016)
ID: 17 | Names: [pillow ,stool ,chair] |  Num embs: 3 | Pcd size: (3, 6543)
ID: 18 | Names: [armchair ,box ,stool ,chair] |  Num embs: 11 | Pcd size: (3, 10340)
ID: 19 | Names: [stool ,armchair ,chair] |  Num embs: 5 | Pcd size: (3, 8121)
ID: 20 | Names: [table] |  Num embs: 1 | Pcd size: (3, 29586)
ID: 21 | Names: [box] |  Num embs: 1 | Pcd size: (3, 1034)
ID: 22 | Names: [armchair] |  Num embs: 3 | Pcd size: (3, 2872)
ID: 23 | Names: [box] |  Num embs: 1 | Pcd size: (3, 3789)


	Localizing image 1 currently
caption post ram:  armchair  .  handcart  .  bin  .  chair  .  container  .  stool  
Phrases:  ['armchair', 'handcart']
[[ 0.95736253  0.7741513   0.8023354   0.8738332   0.39538857  0.8460262
   0.87247384  0.68519604  0.23706533  0.05320997  0.3121214   0.00735207
   0.6368605   0.12615763 -0.2194957   0.5502488   0.09799755  0.13182943
   0.5947074   0.44806504  0.4164306   0.62098074 -0.00613015  0.6003202 ]
 [ 0.77994555  0.9814048   0.6413524   0.8966618   0.09717628  0.880713
   0.83765376  0.67270446  0.3366208  -0.13136482  0.26403475  0.02832221
   0.7937167   0.01412915 -0.38491958  0.60857177 -0.09549006 -0.0453226
   0.3377287   0.3195543   0.37558568  0.41773736 -0.12362048  0.69196874]]
Assignments being considered:  [[[0, 14]], [[0, 22]], [[1, 1]], [[0, 0]], [[1, 3]]]




[[[0, 0]], array([[ 9.99998904e-01,  8.85941695e-04,  1.18657550e-03,
        -7.18130565e-03],
       [-1.73116034e-04,  8.65738153e-01, -5.00497174e-01,
        -4.62104990e-02],
       [-1.47067499e-03,  5.00496420e-01,  8.65737357e-01,
        -5.49528351e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.003977636647809305]
Target pose:  [ 3.75000000e+00  9.00999784e-01  2.02500000e+01  2.58818961e-01
  9.15674411e-04 -2.45354252e-04  9.65925384e-01]
Estimated pose:  [ 5.58751575e+00  3.86297246e-01  2.20479318e+01  2.59096215e-01
  6.87800150e-04 -2.74125479e-04  9.65851232e-01]
Translation error:  2.621820295262706
Rotation_error:  0.00036759022643410903

	Localizing image 11 currently
caption post ram:  barrel  .  bin  .  chair  .  container  .  plastic  .  recycling bin  .  stool  
Phrases:  ['barrel', 'chair']
[[ 0.814767    0.9780214   0.67407775  0.89207333  0.10987598  0.871325
   0.849515    0.713852    0.2934895  -0.12846847  0.22600347  0.04605637
   0.79810417  0.01623336 -0.37194526  0.662972   -0.07387792 -0.02642833
   0.40972218  0.39047724  0.40677994  0.49078107 -0.11602022  0.71991503]
 [ 0.8331471   0.47307366  0.8117466   0.57247794  0.21855071  0.5245879
   0.62934434  0.5265841   0.12762281  0.36652377  0.34761414 -0.08800206
   0.41315204  0.19678608 -0.05341908  0.48214048  0.21821532  0.35760576
   0.74214363  0.4330144   0.4080201   0.656094    0.05882167  0.537782  ]]
Assignments being considered:  [[[0, 1]], [[0, 1], [1, 0]], [[0, 3]], [[0, 1], [1, 2]], [[0, 5]]]



[[[0, 1]], array([[ 9.90698295e-01,  6.83509467e-02,  1.17664932e-01,
         3.59119257e-03],
       [-3.17997216e-04,  8.65855732e-01, -5.00293663e-01,
         5.90704944e-02],
       [-1.36076401e-01,  4.95602662e-01,  8.57823533e-01,
        -9.86451537e-03],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.0040248611190798]
Target pose:  [ 5.00000000e+00  9.00999784e-01  2.05000000e+01  2.58233088e-01
  6.49638172e-02 -1.74070070e-02  9.63738746e-01]
Estimated pose:  [ 3.83290660e+00  9.92031797e-01  2.35159904e+01  2.58369420e-01
  6.58291425e-02 -1.78150625e-02  9.63636026e-01]
Translation error:  3.2352113086682275
Rotation_error:  0.000971820567562003

	Localizing image 21 currently
caption post ram:  armchair  .  chair  .  stool  
Phrases:  ['armchair']
[[ 0.9469627   0.8476794   0.84593284  0.9109906   0.14619839  0.86629736
   0.9028307   0.6901984   0.27134448  0.00701564  0.35203555 -0.01549757
   0.7000952   0.07523058 -0.2672148   0.62428457 -0.05895726  0.10721648
   0.5670712   0.4169158   0.46887058  0.58687234 -0.12530296  0.6074869 ]]
Assignments being considered:  [[[0, 0]], [[0, 3]], [[0, 6]], [[0, 5]], [[0, 1]]]





[[[0, 0]], array([[ 8.12077780e-01,  2.92647512e-01,  5.04863460e-01,
         1.46611722e-02],
       [-1.32028093e-04,  8.65252164e-01, -5.01336888e-01,
        -1.41511461e-02],
       [-5.83549194e-01,  4.07057891e-01,  7.02690694e-01,
        -5.58017913e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.004181444410614216]
Target pose:  [ 6.75        0.90099978 20.75        0.2463329   0.29639666 -0.07941926
  0.91932677]
Estimated pose:  [ 5.33880431  0.7362584  18.96678276  0.24705006  0.29600831 -0.07962529
  0.91924162]
Translation error:  2.2800168225857274
Rotation_error:  0.0008454776984349227

	Localizing image 31 currently
caption post ram:  armchair  .  chair  .  stool  
Phrases:  ['armchair', 'armchair']
[[ 0.98170066  0.8045076   0.82775086  0.90754664  0.33080724  0.8848797
   0.90683085  0.69013166  0.23517467  0.00819629  0.3240763   0.0061846
   0.6633624   0.09040678 -0.22171158  0.59308964 -0.0011351   0.12366223
   0.58999944  0.43878275  0.4694675   0.627703   -0.09110739  0.59245485]
 [ 0.7429326   0.9330178   0.5858608   0.8578895   0.24729571  0.83959347
   0.80730283  0.559687    0.2718937  -0.2034128   0.16723055  0.00484063
   0.7169622  -0.04255394 -0.3653823   0.48473415 -0.11279693 -0.14217722
   0.2777496   0.20201492  0.26884604  0.31086272 -0.12997603  0.67245555]]
Assignments being considered:  [[[0, 0]], [[0, 6]], [[0, 5]], [[0, 14]], [[0, 16]]]





[[[0, 0]], array([[ 9.70797357e-01,  1.19803530e-01,  2.07845148e-01,
        -1.83960991e-02],
       [ 1.28031045e-04,  8.66119965e-01, -4.99836163e-01,
        -1.90171962e-02],
       [-2.39900969e-01,  4.85266237e-01,  8.40811635e-01,
        -5.99681529e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.004011970718020943]
Target pose:  [ 4.5         0.90099978 21.5         0.25691604  0.11691894 -0.03132834
  0.95882357]
Estimated pose:  [ 4.18354959  0.94208986 16.56179058  0.25683933  0.11673793 -0.03120221
  0.95887029]
Translation error:  4.94850902123997
Rotation_error:  0.00023820167591546718

	Localizing image 41 currently
caption post ram:  armchair  .  lamp  .  plaster bandage  .  chair  .  stool  
Phrases:  ['armchair', 'plaster bandage']
[[ 9.77412820e-01  7.75795817e-01  8.48690391e-01  8.63055229e-01
   2.72724807e-01  8.29254448e-01  8.80602896e-01  7.00756431e-01
   1.93450481e-01  2.60141287e-02  3.17606390e-01 -9.82638448e-05
   6.69991970e-01  9.24858004e-02 -2.15160459e-01  6.45643473e-01
  -1.19965812e-02  1.51016325e-01  6.43649817e-01  4.73629504e-01
   5.00379205e-01  6.70241416e-01 -1.06193386e-01  6.08507335e-01]
 [ 8.57164264e-01  6.00179613e-01  9.79182005e-01  7.36239552e-01
   2.42939383e-01  6.84038877e-01  8.22020113e-01  4.68558609e-01
   1.87465444e-01  1.49968579e-01  2.52690524e-01 -5.95199913e-02
   4.45098579e-01  6.62041232e-02 -7.84822255e-02  4.79509473e-01
   1.30850691e-02  1.85661256e-01  6.76589251e-01  3.57312649e-01
   4.27284390e-01  5.57277560e-01 -9.81146470e-02  5.07753313e-01]]
Assignments being considered:  [[[0, 0]], [[0, 6]], [[0, 11]], [[0, 14]], [[0, 16]]]





[[[0, 0]], array([[ 8.80471946e-01,  2.36510889e-01,  4.10891411e-01,
        -4.00173049e-02],
       [ 6.36128110e-04,  8.66089737e-01, -4.99888150e-01,
         1.45460993e-02],
       [-4.74097825e-01,  4.40398872e-01,  7.62417265e-01,
        -5.20152568e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.0039647550748071305]
Target pose:  [ 4.75        0.90099978 22.25        0.25093731  0.2365601  -0.0633861
  0.93651054]
Estimated pose:  [ 4.43429943  0.94170524 17.31161886  0.25098055  0.23622051 -0.06295947
  0.93661344]
Translation error:  4.948629316958622
Rotation_error:  0.000556588592329379

	Localizing image 51 currently
caption post ram:  armchair  .  lamp  .  chair  
Phrases:  ['armchair']
[[ 0.9479252   0.82722384  0.82072634  0.9421319   0.25273198  0.90409815
   0.93776596  0.64013296  0.23759228 -0.04475756  0.32346    -0.01611635
   0.6480824   0.03617241 -0.23749746  0.5547484  -0.1278095   0.0497174
   0.5274483   0.37181658  0.43831012  0.53035504 -0.1790738   0.52318406]]
Assignments being considered:  [[[0, 0]], [[0, 3]], [[0, 6]], [[0, 5]], [[0, 1]]]





[[[0, 0]], array([[ 7.20802649e-01,  3.47194256e-01,  5.99916402e-01,
        -4.60769299e-04],
       [-3.44673522e-04,  8.65684381e-01, -5.00590085e-01,
         1.98743508e-02],
       [-6.93140262e-01,  3.60619884e-01,  6.24107264e-01,
        -7.49395222e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.004088800369522619]
Target pose:  [ 6.25        0.90099978 21.75        0.23997695  0.36180445 -0.09694522
  0.89560607]
Estimated pose:  [ 4.83898541  0.7345352  19.96548891  0.24031802  0.36082353 -0.09697968
  0.89590657]
Translation error:  2.2810419718195076
Rotation_error:  0.001081673222434685

	Localizing image 61 currently
caption post ram:   stand 
Phrases:  ['stand']
[[ 0.92177224  0.8159634   0.83324003  0.9483202   0.26148558  0.94322824
   0.9736952   0.68935996  0.22055797 -0.05390273  0.28476822  0.04853361
   0.67749244  0.06130277 -0.175538    0.6579542  -0.08832957  0.11863375
   0.58554006  0.47445196  0.55369616  0.6111516  -0.15624708  0.53070474]]
Assignments being considered:  [[[0, 6]], [[0, 3]], [[0, 5]], [[0, 0]], [[0, 2]]]





[[[0, 6]], array([[ 9.77920461e-01,  1.04458758e-01,  1.80997073e-01,
        -1.84972665e-01],
       [ 1.20446553e-04,  8.65826291e-01, -5.00344700e-01,
         1.07051531e-04],
       [-2.08977410e-01,  4.89319120e-01,  8.46696664e-01,
         7.78177616e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.014114511240211942]
Target pose:  [ 3.75        0.90099978 22.75        0.25440954  0.17754135 -0.04757207
  0.9494692 ]
Estimated pose:  [ 3.70097126  0.90152945 22.2123507   0.25758373  0.1015002  -0.02715655
  0.96052634]
Translation error:  0.5398804211702143
Rotation_error:  0.07959100717634206

	Localizing image 71 currently
caption post ram:  armchair  .  chair  .  stool  
Phrases:  ['armchair']
[[ 0.9528961   0.84442437  0.7905706   0.9390195   0.31773242  0.91210806
   0.92290413  0.6882325   0.25251377 -0.05189375  0.31995335  0.03088203
   0.6837735   0.07868868 -0.22920057  0.5821719  -0.04963499  0.07887791
   0.54054534  0.40770984  0.46912837  0.5764464  -0.11417384  0.56875324]]
Assignments being considered:  [[[0, 0]], [[0, 3]], [[0, 6]], [[0, 5]], [[0, 1]]]





[[[0, 0]], array([[ 6.68478651e-01,  3.71141755e-01,  6.44507634e-01,
        -3.10013463e-02],
       [ 2.21903639e-04,  8.66487068e-01, -4.99199470e-01,
         3.40561267e-02],
       [-7.43731298e-01,  3.33847207e-01,  5.79145749e-01,
        -5.63274795e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.0039733466293044055]
Target pose:  [ 5.5         0.90099978 22.75        0.23632976  0.39382524 -0.10552517
  0.88199454]
Estimated pose:  [ 4.08581269  0.73281684 20.96385285  0.23603275  0.39333912 -0.10509523
  0.88234226]
Translation error:  2.2844108407691173
Rotation_error:  0.0007939044790742362

	Localizing image 81 currently
caption post ram:  lamp  .  chair  .  stool  
Phrases:  ['lamp']
[[ 0.94708556  0.8543959   0.80051464  0.95167494  0.30603203  0.92657924
   0.9448405   0.7042434   0.25780833 -0.07185189  0.28965503  0.05642345
   0.6931375   0.06445979 -0.22148712  0.6072148  -0.06537091  0.05772799
   0.5536666   0.43862286  0.48721775  0.5809339  -0.12444355  0.5592258 ]]
Assignments being considered:  [[[0, 3]], [[0, 0]], [[0, 6]], [[0, 5]], [[0, 1]]]





[[[0, 0]], array([[ 5.83110768e-01,  4.05510501e-01,  7.03948198e-01,
        -3.31303814e-02],
       [ 4.60896321e-04,  8.66346990e-01, -4.99442369e-01,
         6.75762431e-02],
       [-8.12392528e-01,  2.91554670e-01,  5.04989361e-01,
        -5.56395224e-02],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]), 0.00397676125551154]
Target pose:  [ 5.75        0.90099978 23.          0.2301735   0.44170004 -0.11835319
  0.85901907]
Estimated pose:  [ 4.3356575   0.73347001 21.214021    0.23009477  0.4410915  -0.11782572
  0.85942526]
Translation error:  2.2843274502852506
Rotation_error:  0.0009053870981303068

	Localizing image 91 currently
caption post ram:  lamp  .  plaster bandage  .  chair  .  stool  
Phrases:  ['lamp']
[[ 0.93446195  0.8549724   0.7675145   0.9502458   0.3087718   0.9347393
   0.9509197   0.76465374  0.22543937 -0.11226466  0.2734243   0.0812773
   0.7198745   0.0790823  -0.22560273  0.6785209  -0.06363774  0.07629472
   0.57292306  0.4923495   0.5404151   0.63093567 -0.12054613  0.5660489 ]]
Assignments being considered:  [[[0, 6]], [[0, 3]], [[0, 5]], [[0, 0]], [[0, 1]]]





[[[0, 0]], array([[ 0.58010082,  0.39203486,  0.713997  , -0.10423939],
       [-0.03067902,  0.88645342, -0.46179987,  0.06324898],
       [-0.81396673,  0.24598576,  0.5262596 ,  0.0877145 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), 0.004312689030112931]
Target pose:  [ 5.          0.90099978 23.75        0.22983601  0.44414124 -0.11900732
  0.85775942]
Estimated pose:  [ 3.55875864  0.72916741 21.8886274   0.20456527  0.44161437 -0.1221734
  0.86498755]
Translation error:  2.3603836473713766
Rotation_error:  0.026595241813680653

