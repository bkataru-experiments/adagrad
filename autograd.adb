with Ada.Text_IO; use Ada.Text_IO;
with Ada.Unchecked_Deallocation;

package body Autograd is

   -- Utilities --

   procedure Free_List is new Ada.Unchecked_Deallocation (Object => Value, Name => Value_Access);

   procedure Build_Topo (V : in out Value; Topo : in out Value_Access);

   procedure Build_Topo (V : in out Value; Topo : in out Value_Access) is
   begin
      if not V.Visited then
         V.Visited := True;
         if V.Prev /= null then
            Build_Topo (V.Prev.all, Topo);
         end if;
         if V.Next /= null then
            Build_Topo (V.Next.all, Topo);
         end if;

         -- insert at head of topological list
         declare
            New_Node : constant Value_Access := new Value'(V);
         begin
            New_Node.Next := Topo;
            Topo := New_Node;
         end;
      end if;
   end Build_Topo;

   -- Constructors --

   function New_Constant (Data : Real) return Value is
      V : Value;
   begin
      V.Data := Data;
      V.Grad := 0.0;
      return V;
   end New_Constant;

   function New_Variable (Data : Real) return Value is
   begin
      return New_Constant (Data);  -- gradient starts at 0
   end New_Variable;

   -- Accessors --

   function Data (V : Value) return Real is
   begin
      return V.Data;
   end Data;

   procedure Set_Data (V : in out Value; Data : Real) is
   begin
      V.Data := Data;
   end Set_Data;

   function Grad (V : Value) return Real is
   begin
      return V.Grad;
   end Grad;

   procedure Set_Grad (V : in out Value; Grad : Real) is
   begin
      V.Grad := Grad;
   end Set_Grad;

   procedure Zero_Grad (V : in out Value) is
   begin
      V.Grad := 0.0;
   end Zero_Grad;

   -- Backward --

   procedure Backward (V : in out Value) is
      Topo : Value_Access := null;
   begin
      -- reset visited flags
      V.Visited := False;
      Build_Topo(V, Topo);

      -- seed gradient
      V.Grad := 1.0;

      -- propagate in reverse topological order
      while Topo /= null loop
         if Topo.Backward /= null then
            Topo.Backward (Topo.all);
         end if;
         Topo := Topo.Next;
      end loop;
   end Backward;

   -- Backward impls --

   procedure Backward_Add (V : in out Value) is
   begin
      V.Prev.Grad := V.Prev.Grad + V.Grad;
      V.Next.Grad := V.Next.Grad + V.Grad;
   end Backward_Add;

   procedure Backward_Sub (V : in out Value) is
   begin
      V.Prev.Grad := V.Prev.Grad + V.Grad;
      V.Next.Grad := V.Next.Grad - V.Grad;
   end Backward_Sub;

   procedure Backward_Mul (V : in out Value) is
   begin
      V.Prev.Grad := V.Prev.Grad + V.Next.Data * V.Grad;
      V.Next.Grad := V.Next.Grad + V.Prev.Data * V.Grad;
   end Backward_Mul;

   procedure Backward_Div (V : in out Value) is
      Inv : constant Real := 1.0 / V.Next.Data;
   begin
      V.Prev.Grad := V.Prev.Grad + Inv * V.Grad;
      V.Next.Grad := V.Next.Grad - V.Prev.Data * Inv * Inv * V.Grad;
   end Backward_Div;

   procedure Backward_Pow (V : in out Value) is
   begin
      V.Prev.Grad := V.Prev.Grad +
         (V.Data * Real (V.Data) ** (Real (V.Data) - 1.0)) * V.Grad;
   end Backward_Pow;

   procedure Backward_ReLU (V : in out Value) is
   begin
      if V.Prev.Data > 0.0 then
         V.Prev.Grad := V.Prev.Grad + V.Grad;
      end if;
   end Backward_ReLU;

   procedure Backward_Sigmoid (V : in out Value) is
      S : constant Real := 1.0 / (1.0 + Real (Ada.Numerics.Elementary_Functions.Exp (-V.Prev.Data)));
   begin
      V.Prev.Grad := V.Prev.Grad + S * (1.0 - S) * V.Grad;
   end Backward_Sigmoid;

   procedure Backward_Tanh (V : in out Value) is
      T : constant Real := V.Data;
   begin
      V.Prev.Grad := V.Prev.Grad + (1.0 - T * T) * V.Grad;
   end Backward_Tanh;

   -- Operator Overloads --

   function "+" (L, R : Value) return Value is
      V : Value;
   begin
      V.Data := L.Data + R.Data;
      V.Prev := new Value'(L);
      V.Next := new Value'(R);
      V.Backward := new Backward_Procedure'(Backward_Add);
      return V;
   end "+";

   function "-" (L, R : Value) return Value is
      V : Value;
   begin
      V.Data := L.Data - R.Data;
      V.Prev := new Value'(L);
      V.Next := new Value'(R);
      V.Backward := new Backward_Procedure'(Backward_Sub);
      return V;
   end "-";

   function "*" (L, R : Value) return Value is
      V : Value;
   begin
      V.Data := L.Data * R.Data;
      V.Prev := new Value'(L);
      V.Next := new Value'(R);
      V.Backward := new Backward_Procedure'(Backward_Mul);
      return V;
   end "*";

   function "/" (L, R : Value) return Value is
      V : Value;
   begin
      V.Data := L.Data / R.Data;
      V.Prev := new Value'(L);
      V.Next := new Value'(R);
      V.Backward := new Backward_Procedure'(Backward_Div);
      return V;
   end "/";

   function "**" (L : Value; R : Real) return Value is
      V : Value;
   begin
      V.Data := L.Data ** R;
      V.Prev := new Value'(L);
      V.Next := null;
      V.Backward := new Backward_Procedure'(Backward_Pow);
      return V;
   end "**";
   
end Autograd;
