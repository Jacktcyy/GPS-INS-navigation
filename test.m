% 清空环境变量 elm
clc
clear
close all
format compact
%% 读取数据
data=xlsread('wine.xls');%3分类
input=data(:,1:end-1);
output=data(:,end);
%% 选择测试集与训练集 随机选择100组作为训练讯据   78组作为测试数据
[m n]=sort(rand(1,178));
input_train=input(n(1:100),:)';
input_test=input(n(101:178),:)';
label_train=output(n(1:100),:)';
label_test=output(n(101:178),:)';
output_train=zeros(3,100);
output_test=zeros(3,78);
for i=1:100
    output_train(label_train(i),i)=1;
    
end
for i=1:78
output_test(label_test(i),i)=1;
end
%归一化
[inputn_train,inputps]=mapminmax(input_train);
[inputn_test,inputtestps]=mapminmax('apply',input_test,inputps);
%% 没有优化的ELM
activation='sig';
TTYPE=1;
[IW,B,LW,TF,TYPE] = elmtrain(inputn_train,label_train,5,activation,1);
%% ELM仿真测试
Tn_sim = elmpredict(inputn_test,IW,B,LW,TF,TYPE);
test_accuracy=(sum(label_test==Tn_sim))/length(label_test)
stem(label_test,'*')
hold on
plot(Tn_sim,'p')
title('没有优化的ELM')
legend('期望输出','实际输出')
xlabel('样本数')
ylabel('类别标签')
%% 节点个数
inputnum=size(input_train,1);
hiddennum=5;
 %[bestchrom,trace]=gaforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%遗传算法
%  [bestchrom,trace]=psoforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%粒子群算法
% [bestchrom,trace]=batforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%蝙蝠算法
%  [bestchrom,trace]=saforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%模拟退火
[bestchrom,trace]=antforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%蚁群算法
% [bestchrom,trace]=afforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%鱼群算法

%% 优化后结果分析
 figure
[r c]=size(trace);
plot(trace,'b--');
title('适应度曲线图')
xlabel('进化代数');ylabel('诊断正确率');
x=bestchrom;
%% 把最优初始阀值权值赋予ELM重新训练与预测
TYPE=1;
if TYPE  == 1
    T1  = ind2vec(label_train);
end
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum)';
%% train
W=reshape(w1,hiddennum,inputnum);
Q=size(inputn_train,2);
BiasMatrix = repmat(B1,1,Q);
tempH = W * inputn_train + BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
LW = pinv(H') * T1';
%% test
T2=ind2vec(label_test);
Q=size(inputn_test,2);
BiasMatrix1 = repmat(B1,1,Q);
tempH1 = W * inputn_test + BiasMatrix1;
H1 = 1 ./ (1 + exp(-tempH1));
TY1=(H1'*LW)';
if TYPE  == 1
    temp_Y1=zeros(1,size(TY1,2));
for n=1:size(TY1,2)
    [max_Y,index]=max(TY1(:,n));
    temp_Y1(n)=index;
end
Y_train=temp_Y1;
end
youhua_test_accuracy=sum(Y_train==label_test)/length(label_test)
figure
stem(label_test,'*')
hold on
plot(Y_train,'p')
title('优化后的ELM')
legend('期望输出','实际输出')