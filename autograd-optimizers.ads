with Autograd; use Autograd;

package Autograd_Optimizers is

   type Optimizer is interface;

   procedure Step (Opt : in out Optimizer) is abstract;

   -- Stochastic Gradient Descent
   type SGD is new Optimizer with private;
   procedure Step (Opt : in out SGD);
   function New_SGD (Learning_Rate : Real) return SGD;

   -- Adam
   type Adam is new Optimizer with private;
   procedure Step (Opt : in out Adam);
   function New_Adam
      (Learning_Rate : Real := 0.001;
       Beta1         : Real := 0.9;
       Beta2         : Real := 0.999;
       Epsilon       : Real := 1.0E-8) return Adam;

   -- Parameters that the optimiser must track
   type Parameter_List is array (Positive range <>) of access Value;

   procedure Add_Parameter (Opt : in out Optimizer'Class; P : access Value);
   procedure Zero_All_Grads (Opt : in out Optimizer'Class);

private

   type SGD is new Optimizer with record
      Lr : Real;
      Params : Parameter_List_Access;
   end record;

   type Adam is new Optimizer with record
      Lr : Real;
      Beta1 : Real;
      Beta2 : Real;
      Epsilon : Real;
      T : Natural := 0;
      M : Parameter_List_Access; -- first moment
      V : Parameter_List_Access; -- second moment
      Params : Parameter_List_Access;
   end record;

   type Parameter_List_Access is access all Parameter_List;

end Autograd_Optimizers;
