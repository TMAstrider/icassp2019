cd /home/featurize
mkdir fsd18kdataset
cd fsd18kdataset

% test dataset
featurize dataset download 230b5329-2563-438a-8c3b-cfff1a7ecf6b
% meta
featurize dataset download 2678f9ba-f3b1-45a6-b712-959c730f2ff7
% doc
featurize dataset download d6324d10-1eba-4155-9175-c1ed018c19bf
% train dataset
featurize dataset download 053ba62c-b297-4f9a-afa4-d4182ebce085
unzip FSDnoisy18k.audio_test.zip 
unzip FSDnoisy18k.audio_train.zip 
unzip FSDnoisy18k.doc.zip
unzip FSDnoisy18k.meta.zip
