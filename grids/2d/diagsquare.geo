ly = 10;
lc = 1/ly;
Point(1) = {0,0,0,lc};
Point(2) = {1,0,0,lc};
Line(1) = {1,2};

out[] = Extrude {0, 1, 0}{Line{1};Layers{ly}; };
Physical Line(2) = {1, out[0]};
Physical Line(1) = {out[2], -out[3]};
Physical Surface(0) = out[1];
