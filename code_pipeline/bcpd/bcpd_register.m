function [template_transf,corr_target] = bcpd_register(target,template,opt)
%BCPD_REGISTER Summary of this function goes here
%% set 'win=1' if windows
win=1;
%% input files
fnw = sprintf('%s/bcpd_code/win/bcpd.exe',   pwd);
if(win==1) bcpd=fnw; end;%else bcpd=fnm; end;

x = target;
y = template;

%% parameters
% general parameters
omg = num2str(opt.omg);
lmd = num2str(opt.lmd);
bet = num2str(opt.beta);
gma = num2str(opt.gamma);
dist = num2str(opt.dist);
K   ='150';%150
J   ='300';%300

% convergence
c   =num2str(opt.tol);
n   =num2str(opt.max_loops);
N   =num2str(opt.min_loops);


%% execution
prm1=sprintf('-q -w%s -b%s -l%s -g%s -e%s',omg,bet,lmd,gma,dist); % commands for general parameters
prm2=sprintf('-c%s -n%s -N%s ',c,n,N); % convergence
if(opt.flag_acc == 0)
    disp('No acceleration')
    cmd =sprintf('%s -x%s -y%s %s %s -syxuvceaT',bcpd,x,y,prm1,prm2); %-sA : to ouput all files
    disp(cmd)
else
    disp('Acceleration with standard parameters')
    cmd =sprintf('%s -x%s -y%s %s %s -syxuvceaT -A',bcpd,x,y,prm1,prm2); %-sA : to ouput all files. -A for acceleration with standard parameters
end    

system(cmd); 

template_transf = readmatrix('output_y.txt');
non_outlier_labels = readmatrix('output_c.txt');
matched_points = readmatrix('output_e.txt');
corr_target = matched_points(:,2);
corr_target(non_outlier_labels(:,2)==0) = nan;
end

