% stochastic simulation of Benjamin node - Heun method
% finding mean escape time for a variety of beta values
% Strongly connected only
%close all
clear
 
%% set up variables

h = 1e-3;                 % time step

alpha = 0.05;             % noise amplitude
lambda = 0.8;             % excitability
nu = 1-lambda;
omega = 0; 

bnum = 20;                 % number of graph points to generate.
betavals = logspace(-3,2, bnum); % 20 log spaced points from 10^-3 and 10^2

kmax=1000;                   % total number of simulations for each beta value to compute mean.

first_E = zeros(1,bnum);
second_E = zeros(1,bnum);
both_E = zeros(1,bnum);

uslc=sqrt(1-sqrt(lambda));  % position of the unstable limit cycle 

%% set up timing
%initime = cputime;
tic

%% compute strongly coupled times

for j=19:bnum% find mean escape time for each beta value
    beta=betavals(j);
    
    Ef=zeros(1,kmax);     % set up escape time vector
    Es=zeros(1,kmax);     % set up escape time vector
    Eb=zeros(1,kmax);     % set up escape time vector
    
    for k=1:kmax
        
        y1 = 0;           % initial values
        y2 = 0;

        n = 1;            % initial index
        t = 0;            % initial time
         
        while 1           % no maximum time
                    
            % evaluate slope of deterministic bit left side of interval
            f1l=(lambda-1+1i*omega)*y1 + ...
                2*y1*(abs(y1)^2) - y1*(abs(y1)^4)+ ...
                beta*(y2 - y1);
            
            f2l=(lambda-1+1i*omega)*y2 + ...
                2*y2*(abs(y2)^2) - y2*(abs(y2)^4)+ ...
                beta*(y1 - y2);
            
            % prediction euler's step
            y1bar = y1 + h*f1l + alpha*sqrt(h)*(randn+sqrt(-1)*randn);
            
            y2bar = y2 + h*f2l + alpha*sqrt(h)*(randn+sqrt(-1)*randn);
            
            % correction
            f1r = (lambda-1+1i*omega)*y1bar + ...
                2*y1bar*(abs(y1bar)^2) - y1bar*(abs(y1bar)^4)+ ...
                beta*(y2bar - y1bar);
            
            f2r = (lambda-1+1i*omega)*y2bar + ...
                2*y2bar*(abs(y2bar)^2) - y2bar*(abs(y2bar)^4)+ ...
                beta*(y1bar - y2bar);
            
            % next y1 and y2 values
            y1 = y1 + h*(f1l+f1r)/2 ... 
                + alpha*sqrt(h)*(randn+sqrt(-1)*randn);

            y2 = y2 + h*(f2l+f2r)/2 ...  
                + alpha*sqrt(h)*(randn+sqrt(-1)*randn);
            
            t=t+h;     % next time step
            n=n+1;     % iteration counter
            
            % when the trajectory reaches lc the node is deemed to have escaped
            if (abs(y1)>uslc || abs(y2)>uslc) && Ef(k)==0  % first node has escaped
                Ef(k)=t;
            end
            
            if (abs(y1)>uslc && abs(y2)>uslc)  % both nodes have escaped                
                break
            end
            
        end; 
       
        Eb(k)=t;
        Es(k)=Eb(k)-Ef(k);        % escape times
        
        n; % how many iterations did the while loop go around
  
    end
    
    first_E(j) = mean(Ef);
    second_E(j) = mean(Es);
    both_E(j) = mean(Eb);
    
    fprintf('%f done \n',beta);
    
end

%% stop timer
%fintime = cputime;
%fprintf('CPUTIME: %g\n', fintime - initime);
toc
%% plot it all
figure; hold on;

% plot bifurcation lines
plot(ones(2,1).*1.54297E-02,[0; max(both_E)+100],'-','linewidth',1)
plot(ones(2,1).*1.64917E-01,[0; max(both_E)+100],'-','linewidth',1)

% plot the means
plot(betavals,both_E,'k-','linewidth',2)
plot(betavals,first_E,'k--','linewidth',2) 
plot(betavals,second_E,'k:','linewidth',2)

legend('SN','PF','00 -> 11','00 -> 01/10', '01/10 -> 11','Location','NorthWest')
xlabel('\beta')
ylabel('E[\tau]','Rotation',0)
set(gca,'xscale','log');
box on
%set(gca,'yscale','log');
%axis([0.001 100 8 18])
set(gca,'position',[0.09 0.09 0.89 0.89])


%% save it as eps
%pnam=sprintf('ben_2all_Cnoise_%dbetas_%dsims_nu0-2_a0-05_h10-3.eps',bnum,kmax);
%pnam=['ben_heun_2bimean_Cnoise_' num2str(bnum) 'betas_' num2str(kmax) 'sims_o' num2str(omega) '_n' strrep(num2str(nu),'.','-') '_a' strrep(num2str(alpha),'.','-') '_h' strrep(num2str(h),'.','-') '.eps']
%snam='10cmsq'; % export style
%s=hgexport('readstyle',snam); %read the style
%hgexport(gcf,pnam,s);

% save as a fig file
%fnam=['ben_heun_2bimean_Cnoise_16-' num2str(bnum) 'betas_' num2str(kmax) 'sims_o' num2str(omega) '_n' strrep(num2str(nu),'.','-') '_a' strrep(num2str(alpha),'.','-') '_h' strrep(num2str(h),'.','-') '.fig'];
%hgsave(gcf,fnam);
