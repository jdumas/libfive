/*
 *  C API for the libfive standard library
 *
 *  DO NOT EDIT BY HAND!
 *  This file is automatically generated from libfive/stdlib/stdlib.h
 *
 *  It was last generated on 2021-02-20 14:40:48 by user mkeeter
 */
#pragma once
#include "libfive/tree/tree.hpp"

// Header (hand-written in gen_c.py)
struct TreeVec2 {
    libfive::Tree x, y;
};
struct TreeVec3 {
    libfive::Tree x, y, z;
};
typedef libfive::Tree TreeFloat;

// Autogenerated content begins here

////////////////////////////////////////////////////////////////////////////////
// csg
libfive::Tree _union(libfive::Tree, libfive::Tree);
libfive::Tree intersection(libfive::Tree, libfive::Tree);
libfive::Tree inverse(libfive::Tree);
libfive::Tree difference(libfive::Tree, libfive::Tree);
libfive::Tree offset(libfive::Tree, TreeFloat);
libfive::Tree clearance(libfive::Tree, libfive::Tree, TreeFloat);
libfive::Tree shell(libfive::Tree, TreeFloat);
libfive::Tree blend_expt(libfive::Tree, libfive::Tree, TreeFloat);
libfive::Tree blend_expt_unit(libfive::Tree, libfive::Tree, TreeFloat);
libfive::Tree blend_rough(libfive::Tree, libfive::Tree, TreeFloat);
libfive::Tree blend_difference(libfive::Tree, libfive::Tree, TreeFloat, TreeFloat);
libfive::Tree morph(libfive::Tree, libfive::Tree, TreeFloat);
libfive::Tree loft(libfive::Tree, libfive::Tree, TreeFloat, TreeFloat);
libfive::Tree loft_between(libfive::Tree, libfive::Tree, TreeVec3, TreeVec3);
////////////////////////////////////////////////////////////////////////////////
// shapes
libfive::Tree circle(TreeFloat, TreeVec2);
libfive::Tree ring(TreeFloat, TreeFloat, TreeVec2);
libfive::Tree polygon(TreeFloat, int, TreeVec2);
libfive::Tree rectangle(TreeVec2, TreeVec2);
libfive::Tree rounded_rectangle(TreeVec2, TreeVec2, TreeFloat);
libfive::Tree rectangle_exact(TreeVec2, TreeVec2);
libfive::Tree rectangle_centered_exact(TreeVec2, TreeVec2);
libfive::Tree triangle(TreeVec2, TreeVec2, TreeVec2);
libfive::Tree box_mitered(TreeVec3, TreeVec3);
libfive::Tree box_mitered_centered(TreeVec3, TreeVec3);
libfive::Tree box_exact_centered(TreeVec3, TreeVec3);
libfive::Tree box_exact(TreeVec3, TreeVec3);
libfive::Tree rounded_box(TreeVec3, TreeVec3, TreeFloat);
libfive::Tree sphere(TreeFloat, TreeVec3);
libfive::Tree half_space(TreeVec3, TreeVec3);
libfive::Tree cylinder_z(TreeFloat, TreeFloat, TreeVec3);
libfive::Tree cone_ang_z(TreeFloat, TreeFloat, TreeVec3);
libfive::Tree cone_z(TreeFloat, TreeFloat, TreeVec3);
libfive::Tree pyramid_z(TreeVec2, TreeVec2, TreeFloat, TreeFloat);
libfive::Tree torus_z(TreeFloat, TreeFloat, TreeVec3);
libfive::Tree gyroid(TreeVec3, TreeFloat);
libfive::Tree array_x(libfive::Tree, int, TreeFloat);
libfive::Tree array_xy(libfive::Tree, int, int, TreeVec2);
libfive::Tree array_xyz(libfive::Tree, int, int, int, TreeVec3);
libfive::Tree array_polar_z(libfive::Tree, int, TreeVec2);
libfive::Tree extrude_z(libfive::Tree, TreeFloat, TreeFloat);
////////////////////////////////////////////////////////////////////////////////
// transforms
libfive::Tree move(libfive::Tree, TreeVec3);
libfive::Tree rotate_z(libfive::Tree, TreeFloat, TreeVec3);
libfive::Tree taper_xy_z(libfive::Tree, TreeVec3, TreeFloat, TreeFloat, TreeFloat);
