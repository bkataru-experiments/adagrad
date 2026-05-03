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


end Autograd;
