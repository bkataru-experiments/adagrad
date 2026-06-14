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

   

end Autograd_Optimizers;
