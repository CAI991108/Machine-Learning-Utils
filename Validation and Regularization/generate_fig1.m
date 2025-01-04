%code used to generate figure 1
load training_data.mat  %load the training data

scatter(x,y)
xlim([-1.5 1.5])
ylim([-0.5 2.5])
hold on
xx = -1.5:0.05:1.5;
plot(xx,xx.^2,'LineWidth',2)
legend('data','target function','FontSize',20)
set(gcf, 'Color', 'white');
set(gca, ...
    'LineWidth' , 1.5                     , ...
    'FontSize'  , 20              , ...
    'FontName'  , 'Times New Roman'         );
xlabel('$x$','FontSize',25,'Interpreter','LaTex');
ylabel('$y$','FontSize',25,'Interpreter','LaTex');