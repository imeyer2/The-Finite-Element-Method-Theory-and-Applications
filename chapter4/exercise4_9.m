% Define the rectangle [-2, 2] x [-2*pi, 2*pi] using legacy format
% Outer rectangle boundary (counter-clockwise orientation)
% r_rect = [
%     2, -2,  2, -2*pi, -2*pi, 1, 0;         % bottom edge
%     2,  2,  2, -2*pi,  2*pi, 1, 0;         % right edge
%     2,  2, -2,  2*pi,  2*pi, 1, 0;         % top edge
%     2, -2, -2,  2*pi, -2*pi, 1, 0;         % left edge
% ];
% 
% dl = [r_rect]';
% 
% [p, e, t] = initmesh(dl, 'hmax', 0.1);
% 
% % Plot the mesh
% pdemesh(p, e, t);
% axis equal
% title('Mesh of Square with Square Hole');
% 




function [area, b, c] = HatGradients(x,y)
    area=polyarea(x,y);
    b=[y(2)-y(3); y(3) - y(1); y(1)-y(2)]/2/area;
    c=[x(2)-x(3); x(3) - x(1); x(1)-x(2)]/2/area;
end


function A = StiffnessAssembler2D(p,t,a)
    np=size(p,2);
    nt=size(t,2);
    A=sparse(np,np);
    for K = 1:nt
        loc2glb = t(1:3,K);
        % print(loc2glb);
        x=p(1,loc2glb);
        y=p(2,loc2glb);
        [area,b,c]=HatGradients(x,y);
        xc = mean(x); yc = mean(y);
        abar = a(xc,xc);
        AK = abar*(b*b'+c*c')*area;
        A(loc2glb,loc2glb) = A(loc2glb, loc2glb) + AK;
    end
end



function R = RobinMassMatrix2D(p,e,kappa)
    np=size(p,2);
    ne=size(e,2);
    R = sparse(np,np);
    for E = 1:ne
        loc2glb = e(1:2, E);
        x = p(1,loc2glb);
        y=p(2,loc2glb);
        len = sqrt((x(1)-x(2))^2 + (y(1)-y(2))^2);
        xc = mean(x); yc = mean(y);
        k = kappa(xc,yc);
        RE = k/6*[2 1; 1 2]*len;
        R(loc2glb, loc2glb) = R(loc2glb, loc2glb) + RE;

    end
end




function r = RobinLoadVector2D(p,e,kappa,gD,gN)
    np=size(p,2);
    ne=size(e,2);
    r=zeros(np,1);
    for E=1:ne
        loc2glb=e(1:2,E);
        x=p(1,loc2glb);
        y=p(2,loc2glb);
        len = sqrt((x(1)-x(2))^2 + (y(1)-y(2))^2);
        xc=mean(x);yc=mean(y);
        tmp = kappa(xc,yc)*gD(xc,yc)+gN(xc,yc);
        rE = tmp*[1; 1]*len/2;
        r(loc2glb) = r(loc2glb) + rE;
    end
end


% 
% function b = LoadAssembler1D(x,f)
%     n = length(x)-1;
%     b = zeros(n+1,1);
%     for i = 1:n
%         h = x(i+1)-x(i);
%         b(i) = b(i) + f(x(i))*h/2;
%         b(i+1) = b(i+1) + f(x(i+1))*h/2;
%     end
% end




function c = ayyy(x,y)
    c = 1;
end


function d = efff(x,y)
    d = 0;
end


function z = gD(x,y)
    z = exp(x)*atan(y);
end

function z = gN(x,y)
    z=0;
end

function z = kappa(x,y)
    z = 10^6;
end







function [R,r] = RobinAssembler2D(p,e,kappa,gD,gN)
    R = RobinMassMatrix2D(p,e,kappa);
    r = RobinLoadVector2D(p,e,kappa,gD,gN);
end



function textbookproblemsolver()
    % Define outer boundary via legacy edge format (type 2 = line segment)
    % Format: [type, x1, x2, y1, y2, left region, right region]
    r_rect = [
        2, -2,  2, -2*pi, -2*pi, 1, 0;   % bottom edge
        2,  2,  2, -2*pi,  2*pi, 1, 0;   % right edge
        2,  2, -2,  2*pi,  2*pi, 1, 0;   % top edge
        2, -2, -2,  2*pi, -2*pi, 1, 0;   % left edge
    ];

    % Feed directly into initmesh
    dl = r_rect';  % transpose to 7x4 matrix for initmesh

    [p, e, t] = initmesh(dl, 'hmax', 0.1);

    % Assemble and solve
    A = StiffnessAssembler2D(p, t, @ayyy);
    [R, r] = RobinAssembler2D(p, e, @kappa, @gD, @gN);
    phi = (A + R) \ r;

    % Plot
    pdeplot(p, e, t, 'XYData', phi, 'ZData', phi, 'Mesh', 'on');
    title('Laplace Solution with Robin BC on [-2, 2] × [-2π, 2π]');
end







textbookproblemsolver()
