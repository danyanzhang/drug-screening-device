classdef DrugSingle
   properties
      Name
      Hill
      EC50
      Emax
      E0
      R
   end
   methods
      function obj = DrugSingle(a,b,c,d)
         if nargin == 4
            obj.Emax = a;
            obj.Hill = b;
            obj.E0 = c;
            obj.EC50 = d;         
         end
      end
   end
end