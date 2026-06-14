with Ada.Unchecked_Deallocation;

package body Autograd_Optimizers is

   -- Helpers

   procedure Free is new Ada.Unchecked_Deallocation
      (Parameter_List, Parameter_List_Access);

   procedure Init_Params (Opt: in out Optimizer'Class) is
   begin
      if Opt.Params = null then
         Opt.Params := new Parameter_List (1 .. 0);
      end if;
   end Init_Params;

   procedure Add_Parameter (Opt : in out Optimizer'Class; P : access Value) is
   begin
      Init_Params (Opt);
      declare
         Old : constant Parameter_List_Access := Opt.Params;
      begin
         Opt.Params := new Parameter_List (1 .. Old'Length + 1);
         for I in Old'Range loop
            Opt.Params (I) := Old (I);
         end loop;
         Opt.Params (Opt.Params'Last) := P;
         Free (Old);
      end;
   end Add_Parameter;

   procedure Zero_All_Grads (Opt : in out Optimizer'Class) is
   begin
      if Opt.Params /= null then
         for I in Opt.Params'Range loop
            Opt.Params (I).Zero_Grad;
         end loop;
      end if;
   end Zero_All_Grads;

   -- SGD --

   function New_SGD (Learning_Rate : Real) return SGD is
   begin
      return (Lr => Learning_Rate, Params => null);
   end New_SGD;

   procedure Step (Opt : in out SGD) is
   begin
      if Opt.Params /= null then
         for I in Opt.Params'Range loop
            declare
               P : Value renames Opt.Params (I).all;
            begin
               P.Set_Data(P.Data - Opt.Lr * P.Grad);
            end;
         end loop;
      end if;
   end Step;

   -- Adam --

   function New_Adam
      (
         Learning_Rate : Real := 0.001;
         Beta1         : Real := 0.9;
         Beta2         : Real := 0.999;
         Epsilon       : Real := 1.0E-8
      ) return Adam
   is
   begin
      return (
         Lr => Learning_Rate,
         Beta1 => Beta1,
         Beta2 => Beta2,
         Epsilon => Epsilon,
         T => 0,
         M => null,
         V => null,
         Params => null
      );
   end New_Adam;

   procedure Step (Opt : in out Adam) is
   begin
      Opt.T := Opt.T + 1;
      if Opt.Params = null then
         return;
      end if;

      -- allocate moment arrays on first use
      if Opt.M = null then
         Opt.M := new Parameter_List (Opt.Params'Range);
         Opt.V := new Parameter_List (Opt.Params'Range);
         for I in Opt.Params'Range loop
            Opt.M (I) := new Value'(New_Constant(0.0));
            Opt.V (I) := new Value'(New_Constant(0.0));
         end loop;
      end if;

      for I in Opt.Params'Range loop
         declare
            P : Value renames Opt.Params (I).all;
            G : constant Real := P.Grad;
            M : Real renames Opt.M (I).Data;
            V : Real renames Opt.V (I).Data;
         begin
            M := Opt.Beta1 * M + (1.0 - Opt.Beta1) * G;
            V := Opt.Beta2 * V + (1.0 - Opt.Beta2) * G * G;
            declare
               M_Hat : constant Real := M / (1.0 - Opt.Beta1 ** Opt.T);
               V_Hat : constant Real := V / (1.0 - Opt.Beta2 ** Opt.T);
            begin
               P.Set_Data (P.Data - Opt.Lr * M_Hat / (Real (Sqrt (V_Hat)) + Opt.Epsilon));
            end;
         end;
      end loop;
   end Step;

end Autograd_Optimizers;
