/*
libfive: a CAD kernel for modeling with implicit functions

Copyright (C) 2021  Matt Keeter

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this file,
You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#include "stdlib_impl.hpp"

using namespace libfive;

#define LIBFIVE_DEFINE_XYZ() const auto x = Tree::X(); (void)x; \
                             const auto y = Tree::Y(); (void)y; \
                             const auto z = Tree::Z(); (void)z; ;
////////////////////////////////////////////////////////////////////////////////
// Operator overloads for C vecs
TreeVec2 operator+(const TreeVec2& a, const TreeVec2& b) {
    return TreeVec2{a.x + b.x, a.y + b.y};
}
TreeVec2 operator-(const TreeVec2& a, const TreeVec2& b) {
    return TreeVec2{a.x - b.x, a.y - b.y};
}
TreeVec2 operator*(const TreeVec2& a, const TreeFloat& b) {
    return TreeVec2{a.x * b, a.y * b};
}
TreeVec2 operator/(const TreeVec2& a, const TreeFloat& b) {
    return TreeVec2{a.x / b, a.y / b};
}
TreeVec3 operator+(const TreeVec3& a, const TreeVec3& b) {
    return TreeVec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
TreeVec3 operator-(const TreeVec3& a, const TreeVec3& b) {
    return TreeVec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
TreeVec3 operator-(const TreeVec3& a) {
    return TreeVec3{-a.x, -a.y, -a.z};
}
TreeVec3 operator*(const TreeVec3& a, const TreeFloat& b) {
    return TreeVec3{a.x * b, a.y * b, a.z * b};
}
TreeVec3 operator/(const TreeVec3& a, const TreeFloat& b) {
    return TreeVec3{a.x / b, a.y / b, a.z / b};
}
////////////////////////////////////////////////////////////////////////////////
// csg
Tree _union(Tree a, Tree b) {
    return min(a, b);
}

Tree intersection(Tree a, Tree b) {
    return max(a, b);
}

Tree inverse(Tree a) {
    return -a;
}

Tree difference(Tree a, Tree b) {
    return intersection(a, inverse(b));
}

Tree offset(Tree a, TreeFloat off) {
    return a - off;
}

Tree clearance(Tree a, Tree b, TreeFloat o) {
    return difference(a, offset(b, o));
}

Tree shell(Tree a, TreeFloat o) {
    return clearance(a, a, o);
}

Tree blend_expt(Tree a, Tree b, TreeFloat m) {
    return -log(exp(-m * a) + exp(-m * b)) / m;
}

Tree blend_expt_unit(Tree a, Tree b, TreeFloat m) {
    return blend_expt(a, b, 2.75 / pow(m, 2));
}

Tree blend_rough(Tree a, Tree b, TreeFloat m) {
    auto c = sqrt(abs(a)) + sqrt(abs(b)) - m;
    return _union(a, _union(b, c));
}

Tree blend_difference(Tree a, Tree b, TreeFloat m, TreeFloat o) {
    return inverse(blend_expt_unit(inverse(a), offset(b, o), m));
}

Tree morph(Tree a, Tree b, TreeFloat m) {
    return a * (1 - m) + b * m;
}

Tree loft(Tree a, Tree b, TreeFloat zmin, TreeFloat zmax) {
    LIBFIVE_DEFINE_XYZ();
    return max(z - zmax, max(zmin - z,
        ((z - zmin) * b + (zmax - z) * a) / (zmax - zmin)));
}

Tree loft_between(Tree a, Tree b, TreeVec3 lower, TreeVec3 upper) {
    LIBFIVE_DEFINE_XYZ();

    const auto f = (z - lower.z) / (upper.z - lower.z);
    const auto g = (upper.z - z) / (upper.z - lower.z);

    a = a.remap(
            x + (f * (lower.x - upper.x)),
            y + (f * (lower.y - upper.y)),
            z);
    b = b.remap(
            x + (g * (upper.x - lower.x)),
            y + (g * (upper.y - lower.y)),
            z);
    return loft(a, b, lower.z, upper.z);
}

////////////////////////////////////////////////////////////////////////////////
// shapes
Tree circle(TreeFloat r, TreeVec2 center) {
    LIBFIVE_DEFINE_XYZ();
    auto c = sqrt(x * x + y * y) - r;
    return move(c, TreeVec3{center.x, center.y, 0});
}

Tree ring(TreeFloat ro, TreeFloat ri, TreeVec2 center) {
    return difference(circle(ro, center), circle(ri, center));
}

Tree polygon(TreeFloat r, int n, TreeVec2 center) {
    LIBFIVE_DEFINE_XYZ();
    r = r * cos(M_PI / n);
    const auto half = y - r;
    auto out = half;
    for (int i=1; i < n; ++i) {
        out = intersection(out, rotate_z(half, 2 * M_PI * i / n, TreeVec3{0, 0, 0}));
    }
    return move(out, TreeVec3{center.x, center.y, 0});
}

Tree rectangle(TreeVec2 a, TreeVec2 b) {
    LIBFIVE_DEFINE_XYZ();
    return max(
        max(a.x - x, x - b.x),
        max(a.y - y, y - b.y));
}

Tree rounded_rectangle(TreeVec2 a, TreeVec2 b, TreeFloat r) {
    return _union(
        _union(rectangle(TreeVec2{a.x, a.y + r}, TreeVec2{b.x, b.y - r}),
               rectangle(TreeVec2{a.x + r, a.y}, TreeVec2{b.x - r, b.y})),
        _union(
            _union(circle(r, TreeVec2{a.x + r, a.y + r}),
                   circle(r, TreeVec2{b.x - r, b.y - r})),
            _union(circle(r, TreeVec2{a.x + r, b.y - r}),
                   circle(r, TreeVec2{b.x - r, a.y + r}))));
}

Tree rectangle_exact(TreeVec2 a, TreeVec2 b) {
    TreeVec2 size = b - a;
    TreeVec2 center = (a + b) / 2;
    return rectangle_centered_exact(size, center);
}

Tree rectangle_centered_exact(TreeVec2 size, TreeVec2 center) {
    LIBFIVE_DEFINE_XYZ();
    const auto dx = abs(x) - size.x/2;
    const auto dy = abs(y) - size.y/2;
    return move(
        min(max(dx, dy), 0) +
        sqrt(square(max(dx, 0)) + square(max(dy, 0))),
        TreeVec3{center.x, center.y, 0});
}

Tree half_plane(TreeVec2 a, TreeVec2 b) {
    LIBFIVE_DEFINE_XYZ();
    return (b.y - a.y) * (x - a.x) - (b.x - a.x) * (y - a.y);
}

Tree triangle(TreeVec2 a, TreeVec2 b, TreeVec2 c) {
    LIBFIVE_DEFINE_XYZ();
    // We don't know which way the triangle is wound, and can't actually
    // know (because it could be parameterized, so we return the union
    // of both possible windings)
    return _union(
        intersection(intersection(
            half_plane(a, b), half_plane(b, c)), half_plane(c, a)),
        intersection(intersection(
            half_plane(a, c), half_plane(c, b)), half_plane(b, a)));
}

//------------------------------------------------------------------------------
Tree box_mitered(TreeVec3 a, TreeVec3 b) {
    return extrude_z(rectangle(TreeVec2{a.x, a.y}, TreeVec2{b.x, b.y}), a.z, b.z);
}

Tree box_mitered_centered(TreeVec3 size, TreeVec3 center) {
    return box_mitered(center - size / 2, center + size / 2);
}

Tree box_exact_centered(TreeVec3 size, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    const auto dx = abs(x - center.x) - (size.x / 2);
    const auto dy = abs(y - center.y) - (size.y / 2);
    const auto dz = abs(z - center.z) - (size.z / 2);
    return min(0, max(dx, max(dy, dz))) + sqrt(square(max(dx, 0)) +
                                               square(max(dy, 0)) +
                                               square(max(dz, 0)));
}

Tree box_exact(TreeVec3 a, TreeVec3 b) {
    return box_exact_centered(b - a, (a + b) / 2);
}

Tree rounded_box(TreeVec3 a, TreeVec3 b, TreeFloat r) {
    const auto d = b - a;
    r = r * min(d.x, min(d.y, d.z)) / 2;
    TreeVec3 v{r, r, r};
    return offset(box_exact(a + v, b - v), r);
}

Tree sphere(TreeFloat r, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    return move(sqrt(square(x) + square(y) + square(z)) - r, center);
}

Tree half_space(TreeVec3 norm, TreeVec3 point) {
    LIBFIVE_DEFINE_XYZ();
    // dot(pos - point, norm)
    return (x - point.x) * norm.x +
           (y - point.y) * norm.y +
           (z - point.z) * norm.z;
}

Tree cylinder_z(TreeFloat r, TreeFloat h, TreeVec3 base) {
    return extrude_z(circle(r, TreeVec2{base.x, base.y}), base.z, base.z + h);
}

Tree cone_ang_z(TreeFloat angle, TreeFloat height, TreeVec3 base) {
    LIBFIVE_DEFINE_XYZ();
    return move(max(-z, cos(angle) * sqrt(square(x) + square(y))
                      + sin(angle) * z - height),
                base);
}

Tree cone_z(TreeFloat radius, TreeFloat height, TreeVec3 base) {
    return cone_ang_z(atan2(radius, height), height, base);
}

Tree pyramid_z(TreeVec2 a, TreeVec2 b, TreeFloat zmin, TreeFloat height) {
    // TODO: make this an intersection of planes instead, to avoid singularity
    return taper_xy_z(
        extrude_z(rectangle(a, b), zmin, zmin + height),
        TreeVec3{ (a.x + b.x) / 2, (a.y + b.y) / 2, zmin},
        height, 0, 1);
}

Tree torus_z(TreeFloat ro, TreeFloat ri, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    return move(
        sqrt(square(ro - sqrt(square(x) + square(y)))
           + square(z)) - ri,
        center);
}

Tree gyroid(TreeVec3 period, TreeFloat thickness) {
    LIBFIVE_DEFINE_XYZ();
    const auto tau = 2 * M_PI;
    return shell(
        sin(x * period.x / tau) * cos(y * period.y / tau) +
        sin(y * period.y / tau) * cos(z * period.z / tau) +
        sin(z * period.z / tau) * cos(x * period.x / tau),
        -thickness);
}

//------------------------------------------------------------------------------

Tree array_x(Tree shape, int nx, TreeFloat dx) {
    for (int i=1; i < nx; ++i) {
        shape = _union(shape, move(shape, TreeVec3{dx * i, 0, 0}));
    }
    return shape;
}

Tree array_xy(Tree shape, int nx, int ny, TreeVec2 delta) {
    shape = array_x(shape, nx, delta.x);
    for (int i=1; i < ny; ++i) {
        shape = _union(shape, move(shape, TreeVec3{0, delta.y * i, 0}));
    }
    return shape;
}

Tree array_xyz(Tree shape, int nx, int ny, int nz,
                         TreeVec3 delta) {
    shape = array_xy(shape, nx, ny, TreeVec2{delta.x, delta.y});
    for (int i=1; i < nz; ++i) {
        shape = _union(shape, move(shape, TreeVec3{0, 0, delta.y * i}));
    }
    return shape;
}

Tree array_polar_z(Tree shape, int n, TreeVec2 center) {
    const float a = 2 * M_PI / n;
    TreeVec3 c{center.x, center.y, 0};
    for (int i=0; i < n; ++i) {
        shape = _union(shape, rotate_z(shape, i * a, c));
    }
    return shape;
}

Tree extrude_z(Tree t, TreeFloat zmin, TreeFloat zmax) {
    LIBFIVE_DEFINE_XYZ();
    return max(t, max(zmin - z, z - zmax));
}


////////////////////////////////////////////////////////////////////////////////
// transforms
Tree move(Tree t, TreeVec3 offset) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x - offset.x, y - offset.y, z - offset.z);
}

Tree reflect_x(Tree t, TreeFloat x0) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(2*x0 - x, y, z);
}

Tree reflect_y(Tree t, TreeFloat y0) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, 2*y0 - y, z);
}

Tree reflect_z(Tree t, TreeFloat z0) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, y, 2*z0 - z);
}

Tree reflect_xy(Tree t) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(y, x, z);
}

Tree reflect_yz(Tree t) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, z, y);
}

Tree reflect_xz(Tree t) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(z, y, x);
}

Tree symmetric_x(Tree t) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(abs(x), y, z);
}

Tree symmetric_y(Tree t) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, abs(y), z);
}

Tree symmetric_z(Tree t) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, y, abs(z));
}

Tree scale_x(Tree t, TreeFloat sx, TreeFloat x0) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x0 + (x - x0) / sx, y, z);
}

Tree scale_y(Tree t, TreeFloat sy, TreeFloat y0) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, y0 + (y - y0) / sy, z);
}

Tree scale_z(Tree t, TreeFloat sz, TreeFloat z0) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(x, y, z0 + (z - z0) / sz);
}

Tree scale_xyz(Tree t, TreeVec3 s, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    return t.remap(
        center.x + (x - center.x) / s.x,
        center.y + (y - center.y) / s.y,
        center.z + (z - center.z) / s.z);
}

Tree rotate_x(Tree t, TreeFloat angle, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    t = move(t, TreeVec3{-center.x, -center.y, -center.z});
    return move(t.remap(x,
                        cos(angle) * y + sin(angle) * z,
                       -sin(angle) * y + cos(angle) * z), center);
}

Tree rotate_y(Tree t, TreeFloat angle, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    t = move(t, TreeVec3{-center.x, -center.y, -center.z});
    return move(t.remap(cos(angle) * x + sin(angle) * z,
                        y,
                       -sin(angle) * x + cos(angle) * z), center);
}

Tree rotate_z(Tree t, TreeFloat angle, TreeVec3 center) {
    LIBFIVE_DEFINE_XYZ();
    t = move(t, TreeVec3{-center.x, -center.y, -center.z});
    return move(t.remap(cos(angle) * x + sin(angle) * y,
                       -sin(angle) * x + cos(angle) * y, z), center);
}

Tree taper_x_y(Tree shape, TreeVec2 base, TreeFloat height,
                TreeFloat scale, TreeFloat base_scale)
{
    LIBFIVE_DEFINE_XYZ();
    const auto s = height / (scale * y + base_scale * (height - y));
    return move(
        move(shape, -TreeVec3{base.x, base.y, 0}).remap(x * s, y, z),
        TreeVec3{base.x, base.y, 0});
}

Tree taper_xy_z(Tree shape, TreeVec3 base, TreeFloat height,
                TreeFloat scale, TreeFloat base_scale)
{
    LIBFIVE_DEFINE_XYZ();
    const auto s = height / (scale * z + base_scale * (height - z));
    return move(
        move(shape, -base).remap(x * s, y * s, z),
        base);
}

Tree shear_x_y(Tree t, TreeVec2 base, TreeFloat height, TreeFloat offset,
               TreeFloat base_offset)
{
    LIBFIVE_DEFINE_XYZ();
    const auto f = (y - base.y) / height;
    return t.remap(x - (base_offset * (1 - f)) - offset * f, y, z);
}

#define AXIS_X 1
#define AXIS_Y 2
#define AXIS_Z 4
Tree attract_repel_generic(Tree shape, TreeVec3 locus,
                           TreeFloat radius, TreeFloat exaggerate, float sign,
                           uint8_t axes)
{
    LIBFIVE_DEFINE_XYZ();
    const auto norm = sqrt(
        square((axes & AXIS_X) ? x : 0) +
        square((axes & AXIS_Y) ? y : 0) +
        square((axes & AXIS_Z) ? z : 0));
    const auto fallout = sign * exaggerate * exp(norm / radius);

    return move(
            move(shape, -locus).remap(
                x * ((axes & AXIS_X) ? fallout : 0),
                y * ((axes & AXIS_Y) ? fallout : 0),
                z * ((axes & AXIS_Z) ? fallout : 0)),
            locus);
}

Tree repel(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(
            shape, locus, radius, exaggerate, -1,
            AXIS_X | AXIS_Y | AXIS_Z);
}

Tree repel_x(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, -1, AXIS_X);
}

Tree repel_y(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, -1, AXIS_Y);
}

Tree repel_z(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, -1, AXIS_Z);
}

Tree repel_xy(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, -1, AXIS_X | AXIS_Y);
}

Tree repel_yz(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, -1, AXIS_Y | AXIS_Z);
}

Tree repel_xz(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, -1, AXIS_X | AXIS_Z);
}

Tree attract(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(
            shape, locus, radius, exaggerate, 1,
            AXIS_X | AXIS_Y | AXIS_Z);
}

Tree attract_x(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, 1, AXIS_X);
}

Tree attract_y(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, 1, AXIS_Y);
}

Tree attract_z(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, 1, AXIS_Z);
}

Tree attract_xy(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, 1, AXIS_X | AXIS_Y);
}

Tree attract_yz(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, 1, AXIS_Y | AXIS_Z);
}

Tree attract_xz(Tree shape, TreeVec3 locus, TreeFloat radius, TreeFloat exaggerate) {
    return attract_repel_generic(shape, locus, radius, exaggerate, 1, AXIS_X | AXIS_Z);
}

Tree revolve_y(Tree shape, TreeFloat x0) {
    LIBFIVE_DEFINE_XYZ();
    const auto r = sqrt(square(x) + square(y));
    const TreeVec3 center{x0, 0, 0};
    shape = move(shape, -center);
    return move(_union(shape.remap(r, y, z), shape.remap(-r, y, z)), center);
}

// This is directly ported from Scheme (hence the use of std::function), and
// I don't totally understand it - refactoring would be welcome.
Tree generic_centered_twirl_x(Tree shape, TreeFloat amount, TreeFloat radius,
                              TreeVec3 vec)
{
    LIBFIVE_DEFINE_XYZ();
    const auto norm = sqrt(square(vec.x) + square(vec.y) + square(vec.z));
    const auto ca = cos(amount * exp(-norm / radius));
    const auto sa = sin(amount * exp(-norm / radius));

    return shape.remap(x,
        ca * y + sa * z,
        ca * z - sa * y);
}

Tree centered_twirl_x(Tree shape, TreeFloat amount, TreeFloat radius) {
    LIBFIVE_DEFINE_XYZ();
    return generic_centered_twirl_x(shape, amount, radius, TreeVec3{x, y, z});
}

Tree centered_twirl_axis_x(Tree shape, TreeFloat amount, TreeFloat radius) {
    LIBFIVE_DEFINE_XYZ();
    return generic_centered_twirl_x(shape, amount, radius, TreeVec3{0, y, z});
}

Tree generic_twirl_n(Tree shape, TreeFloat amount,
                     TreeFloat radius, TreeVec3 center,
                     std::function<Tree(Tree, TreeFloat, TreeFloat)> method,
                     std::function<Tree(Tree)> remap)
{
    shape = move(shape, -center);
    shape = remap(shape);
    shape = method(shape, amount, radius);
    shape = remap(shape);
    return move(shape, center);
}

Tree generic_twirl_x(Tree shape, TreeFloat amount, TreeFloat radius,
                     TreeVec3 center,
                     std::function<Tree(Tree, TreeFloat, TreeFloat)> method)
{
    return generic_twirl_n(shape, amount, radius, center, method,
                           [](Tree t) { return t; });
}

Tree generic_twirl_y(Tree shape, TreeFloat amount, TreeFloat radius,
                     TreeVec3 center,
                     std::function<Tree(Tree, TreeFloat, TreeFloat)> method)
{
    return generic_twirl_n(shape, amount, radius, center, method, reflect_xy);
}

Tree generic_twirl_z(Tree shape, TreeFloat amount, TreeFloat radius,
                     TreeVec3 center,
                     std::function<Tree(Tree, TreeFloat, TreeFloat)> method)
{
    return generic_twirl_n(shape, amount, radius, center, method, reflect_xz);
}

Tree twirl_x(Tree shape, TreeFloat amount, TreeFloat radius, TreeVec3 center) {
    return generic_twirl_x(shape, amount, radius, center, centered_twirl_x);
}
Tree twirl_axis_x(Tree shape, TreeFloat amount, TreeFloat radius, TreeVec3 center) {
    return generic_twirl_x(shape, amount, radius, center, centered_twirl_axis_x);
}

Tree twirl_y(Tree shape, TreeFloat amount, TreeFloat radius, TreeVec3 center) {
    return generic_twirl_y(shape, amount, radius, center, centered_twirl_x);
}
Tree twirl_axis_y(Tree shape, TreeFloat amount, TreeFloat radius, TreeVec3 center) {
    return generic_twirl_y(shape, amount, radius, center, centered_twirl_axis_x);
}

Tree twirl_z(Tree shape, TreeFloat amount, TreeFloat radius, TreeVec3 center) {
    return generic_twirl_z(shape, amount, radius, center, centered_twirl_x);
}
Tree twirl_axis_z(Tree shape, TreeFloat amount, TreeFloat radius, TreeVec3 center) {
    return generic_twirl_z(shape, amount, radius, center, centered_twirl_axis_x);
}
////////////////////////////////////////////////////////////////////////////////
// Text
typedef std::pair<float, Tree> Glyph; // Glyphs are stored as [width, shape]

Glyph glyph_A(void) {
    return Glyph(0.8,
        _union(_union(_union(_union(
           triangle(TreeVec2{0, 0}, TreeVec2{0.35, 1}, TreeVec2{0.1, 0}),
           triangle(TreeVec2{0.1, 0}, TreeVec2{0.35, 1}, TreeVec2{0.45, 1})),
           triangle(TreeVec2{0.35, 1}, TreeVec2{0.45, 1}, TreeVec2{0.8, 0})),
           triangle(TreeVec2{0.7, 0}, TreeVec2{0.35, 1}, TreeVec2{0.8, 0})),
           rectangle(TreeVec2{0.2, 0.3}, TreeVec2{0.6, 0.4})));
}

Glyph glyph_a(void) {
    return Glyph(0.58,
        move(_union(
                shear_x_y(
                    ring(0.275, 0.175, TreeVec2{0.25, 0.275}),
                    TreeVec2{0, 0}, 0.35, 0.1, 0),
                rectangle(TreeVec2{0.51, 0}, TreeVec2{0.61, 0.35})),
            TreeVec3{-0.05, 0, 0}));
}

Glyph glyph_B(void) {
    return Glyph(0.575,
        _union(intersection(_union(ring(0.275, 0.175, TreeVec2{0.3, 0.725}),
                                    ring(0.275, 0.175, TreeVec2{0.3, 0.275})),
                             rectangle(TreeVec2{0.3, 0}, TreeVec2{1, 1})),
                _union(_union(_union(
                     rectangle(TreeVec2{0.1, 0}, TreeVec2{0.3, 0.1}),
                     rectangle(TreeVec2{0.1, 0.45}, TreeVec2{0.3, 0.55})),
                     rectangle(TreeVec2{0.1, 0.9}, TreeVec2{0.3, 1})),
                     // Main bar of the letter
                     rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}))));
}

Glyph glyph_b(void) {
    return Glyph(0.525,
      _union(intersection(ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
                          rectangle(TreeVec2{0.275, 0}, TreeVec2{1, 1})),
            _union(_union(
                 rectangle(TreeVec2{0.1, 0}, TreeVec2{0.275, 0.1}),
                 rectangle(TreeVec2{0.1, 0.45}, TreeVec2{0.275, 0.55})),
                 rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}))));
}

Glyph glyph_C(void) {
    return Glyph(0.57,
        _union(difference(
                _union(ring(0.275, 0.175, TreeVec2{0.3, 0.7}),
                       ring(0.275, 0.175, TreeVec2{0.3, 0.3})),
                _union(rectangle(TreeVec2{0, 0.3}, TreeVec2{0.6, 0.7}),
                       triangle(TreeVec2{0.3, 0.5}, TreeVec2{1, 1.5},
                                TreeVec2{1, -0.5}))),
           rectangle(TreeVec2{0.025, 0.3}, TreeVec2{0.125, 0.7})));
}

Glyph glyph_c(void) {
    return Glyph(0.48,
        difference(ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
                   triangle(TreeVec2{0.275, 0.275}, TreeVec2{0.55, 0.555},
                            TreeVec2{0.55, 0})));
}

Glyph glyph_D(void) {
    return Glyph(0.6,
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
               intersection(rectangle(TreeVec2{0, 0}, TreeVec2{1, 1}),
                            ring(0.5, 0.4, TreeVec2{0.1, 0.5}))));
}

Glyph glyph_d(void) {
    auto g = glyph_b();
    return Glyph(g.first, reflect_x(g.second, g.first/2));
}

Glyph glyph_E(void) {
    return Glyph(0.6,
        _union(_union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
                      rectangle(TreeVec2{0, 0}, TreeVec2{0.6, 0.1})),
               _union(rectangle(TreeVec2{0, 0.9}, TreeVec2{0.6, 1}),
                      rectangle(TreeVec2{0, 0.45}, TreeVec2{0.6, 0.55}))));
}

Glyph glyph_e(void) {
    return Glyph(0.55,
        intersection(
            _union(
                difference(
                    ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
                    triangle(TreeVec2{0.1, 0.275}, TreeVec2{0.75, 0.275},
                             TreeVec2{0.6, 0})),
                rectangle(TreeVec2{0.05, 0.225}, TreeVec2{0.55, 0.315})),
            circle(0.275, TreeVec2{0.275, 0.275})));
}

Glyph glyph_F(void) {
    return Glyph(0.6,
        _union(_union(
            rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
            rectangle(TreeVec2{0, 0.9}, TreeVec2{0.6, 1})),
            rectangle(TreeVec2{0, 0.45}, TreeVec2{0.6, 0.55})));
}

Glyph glyph_f(void) {
    return Glyph(0.4,
        _union(
            intersection(ring(0.25, 0.15, TreeVec2{0.4, 0.75}),
                         rectangle(TreeVec2{0, 0.75}, TreeVec2{0.4, 1})),
            _union(
                rectangle(TreeVec2{0, 0.45}, TreeVec2{0.4, 0.55}),
                rectangle(TreeVec2{0.15, 0.}, TreeVec2{0.25, 0.75}))));
}

Glyph glyph_G(void) {
    return Glyph(0.6,
        _union(
            _union(
                rectangle(TreeVec2{0, 0.3}, TreeVec2{0.1, 0.7}),
                rectangle(TreeVec2{0.5, 0.3}, TreeVec2{0.6, 0.4})),
            _union(
                rectangle(TreeVec2{0.3, 0.4}, TreeVec2{0.6, 0.5}),
                difference(
                    _union(ring(0.3, 0.2, TreeVec2{0.3, 0.7}),
                           ring(0.3, 0.2, TreeVec2{0.3, 0.3})),
                    rectangle(TreeVec2{0, 0.3}, TreeVec2{0.6, 0.7})))));
}

Glyph glyph_g(void) {
    return Glyph(0.55,
        _union(intersection(
                ring(0.275, 0.175, TreeVec2{0.275, -0.1}),
                rectangle(TreeVec2{0, -0.375}, TreeVec2{0.55, -0.1})),
        _union(
            ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
            rectangle(TreeVec2{0.45, -0.1}, TreeVec2{0.55, 0.55}))));
}

Glyph glyph_H(void) {
    return Glyph(0.6,
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
        _union(rectangle(TreeVec2{0.5, 0}, TreeVec2{0.6, 1}),
               rectangle(TreeVec2{0, 0.45}, TreeVec2{0.6, 0.55}))));
}

Glyph glyph_h(void) {
    return Glyph(0.6,
        _union(
            intersection(ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
                         rectangle(TreeVec2{0, 0.275}, TreeVec2{0.55, 0.55})),
            _union(
                rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
                rectangle(TreeVec2{0.45, 0}, TreeVec2{0.55, 0.275}))));
}

Glyph glyph_I(void) {
    return Glyph(0.5,
        _union(rectangle(TreeVec2{0.2, 0}, TreeVec2{0.3, 1}),
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.5, 0.1}),
               rectangle(TreeVec2{0, 0.9}, TreeVec2{0.5, 1}))));
}

Glyph glyph_i(void) {
    return Glyph(0.15,
        _union(circle(0.075, TreeVec2{0.075, 0.7}),
               rectangle(TreeVec2{0.025, 0}, TreeVec2{0.125, 0.55})));
}

Glyph glyph_J(void) {
    return Glyph(0.55,
        _union(
            rectangle(TreeVec2{0.4, 0.275}, TreeVec2{0.5, 1}),
            intersection(
                ring(0.225, 0.125, TreeVec2{0.275, 0.275}),
                rectangle(TreeVec2{0, 0}, TreeVec2{0.55, 0.275}))));
}

Glyph glyph_j(void) {
    return Glyph(0.3,
        _union(_union(
                circle(0.075, TreeVec2{0.225, 0.7}),
                rectangle(TreeVec2{0.175, -0.1}, TreeVec2{0.275, 0.55})),
            intersection(
                ring(0.275, 0.175, TreeVec2{0, -0.1}),
                rectangle(TreeVec2{0, -0.375}, TreeVec2{0.55, -0.1}))));
}

Glyph glyph_K(void) {
    return Glyph(0.6,
        difference(rectangle(TreeVec2{0, 0}, TreeVec2{0.6, 1}),
            _union(_union(
                triangle(TreeVec2{0.1, 1}, TreeVec2{0.5, 1},
                         TreeVec2{0.1, 0.6}),
                triangle(TreeVec2{0.5, 0}, TreeVec2{0.1, 0},
                         TreeVec2{0.1, 0.4})),
                triangle(TreeVec2{0.6, 0.95}, TreeVec2{0.6, 0.05},
                         TreeVec2{0.18, 0.5}))));
}

Glyph glyph_k(void) {
    return Glyph(0.5,
        difference(rectangle(TreeVec2{0, 0}, TreeVec2{0.5, 1}),
            _union(_union(
                triangle(TreeVec2{0.1, 1}, TreeVec2{0.5, 1},
                         TreeVec2{0.1, 0.45}),
                triangle(TreeVec2{0.37, -0.1}, TreeVec2{0.1, -0.1},
                         TreeVec2{0.1, 0.25})),
            _union(
                triangle(TreeVec2{0.6, 1}, TreeVec2{0.5, 0},
                         TreeVec2{0.18, 0.35}),
                triangle(TreeVec2{0.1, 1}, TreeVec2{0.6, 1},
                         TreeVec2{0.6, 0.5})))));
}

Glyph glyph_L(void) {
    return Glyph(0.6,
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
               rectangle(TreeVec2{0, 0}, TreeVec2{0.6, 0.1})));
}

Glyph glyph_l(void) {
    return Glyph(0.15,
        rectangle(TreeVec2{0.025, 0}, TreeVec2{0.125, 1}));
}

Glyph glyph_M(void) {
    return Glyph(0.8,
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
        _union(rectangle(TreeVec2{0.7, 0}, TreeVec2{0.8, 1}),
        _union(triangle(TreeVec2{0, 1}, TreeVec2{0.1, 1}, TreeVec2{0.45, 0}),
        _union(triangle(TreeVec2{0.45, 0}, TreeVec2{0.35, 0}, TreeVec2{0, 1}),
        _union(triangle(TreeVec2{0.7, 1}, TreeVec2{0.8, 1}, TreeVec2{0.35, 0}),
        triangle(TreeVec2{0.35, 0}, TreeVec2{0.8, 1}, TreeVec2{0.45, 0})))))));
}

Glyph glyph_m(void) {
    return Glyph(0.6,
        _union(
            intersection(
                _union(ring(0.175, 0.075, TreeVec2{0.175, 0.35}),
                       ring(0.175, 0.075, TreeVec2{0.425, 0.35})),
                 rectangle(TreeVec2{0, 0.35}, TreeVec2{0.65, 0.65})),
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 0.525}),
        _union(rectangle(TreeVec2{0.25, 0}, TreeVec2{0.35, 0.35}),
        rectangle(TreeVec2{0.5, 0}, TreeVec2{0.6, 0.35})))));
}

Glyph glyph_N(void) {
    return Glyph(0.6,
        _union(_union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
                      rectangle(TreeVec2{0.5, 0}, TreeVec2{0.6, 1})),
        _union(
            triangle(TreeVec2{0, 1}, TreeVec2{0.1, 1}, TreeVec2{0.6, 0}),
            triangle(TreeVec2{0.6, 0}, TreeVec2{0.5, 0}, TreeVec2{0, 1}))));
}

Glyph glyph_n(void) {
    return Glyph(0.55,
        _union(intersection(
                ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
                rectangle(TreeVec2{0, 0.325}, TreeVec2{0.55, 0.55})),
            _union(
                rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 0.55}),
                rectangle(TreeVec2{0.45, 0}, TreeVec2{0.55, 0.325}))));
}

Glyph glyph_O(void) {
    return Glyph(0.6,
        _union(_union(rectangle(TreeVec2{0, 0.3}, TreeVec2{0.1, 0.7}),
                      rectangle(TreeVec2{0.5, 0.3}, TreeVec2{0.6, 0.7})),
            difference(
                _union(ring(0.3, 0.2, TreeVec2{0.3, 0.7}),
                       ring(0.3, 0.2, TreeVec2{0.3, 0.3})),
                rectangle(TreeVec2{0, 0.3}, TreeVec2{0.6, 0.7}))));
}

Glyph glyph_o(void) {
    return Glyph(0.55, ring(0.275, 0.175, TreeVec2{0.275, 0.275}));
}

Glyph glyph_P(void) {
    return Glyph(0.575,
        _union(difference(ring(0.275, 0.175, TreeVec2{0.3, 0.725}),
                          rectangle(TreeVec2{0, 0}, TreeVec2{0.3, 1})),
        _union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1}),
        _union(rectangle(TreeVec2{0.1, 0.45}, TreeVec2{0.3, 0.55}),
        rectangle(TreeVec2{0.1, 0.9}, TreeVec2{0.3, 1})))));
}

Glyph glyph_p(void) {
    return Glyph(0.55,
        _union(ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
               rectangle(TreeVec2{0, -0.375}, TreeVec2{0.1, 0.55})));
}

Glyph glyph_Q(void) {
    return Glyph(0.6,
        _union(rectangle(TreeVec2{0, 0.3}, TreeVec2{0.1, 0.7}),
        _union(rectangle(TreeVec2{0.5, 0.3}, TreeVec2{0.6, 0.7}),
        _union(
            difference(
                _union(ring(0.3, 0.2, TreeVec2{0.3, 0.7}),
                       ring(0.3, 0.2, TreeVec2{0.3, 0.3})),
                 rectangle(TreeVec2{0, 0.3}, TreeVec2{0.6, 0.7})),
        _union(triangle(TreeVec2{0.5, 0.1}, TreeVec2{0.6, 0.1},
                        TreeVec2{0.6, 0}),
               triangle(TreeVec2{0.5, 0.1}, TreeVec2{0.5, 0.3},
                        TreeVec2{0.6, 0.1}))))));
}

Glyph glyph_q(void) {
    return Glyph(0.55,
        _union(ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
               rectangle(TreeVec2{0.45, -0.375}, TreeVec2{0.55, 0.55})));
}

Glyph glyph_R(void) {
    return Glyph(0.575,
        _union(glyph_P().second,
        _union(
            triangle(TreeVec2{0.3, 0.5}, TreeVec2{0.4, 0.5},
                     TreeVec2{0.575, 0}),
            triangle(TreeVec2{0.475, 0}, TreeVec2{0.3, 0.5},
                     TreeVec2{0.575, 0}))));
}

Glyph glyph_r(void) {
    return Glyph(0.358,
        _union(
            scale_x(
                intersection(
                    difference(
                        circle(0.55, TreeVec2{0.55, 0}),
                        scale_x(circle(0.45, TreeVec2{0.55, 0}), 0.8, 0.55)),
                    rectangle(TreeVec2{0, 0}, TreeVec2{0.55, 0.55})), 0.7, 0),
            rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 0.55})));
}

Glyph glyph_S(void) {
    const auto half = difference(
        ring(0.275, 0.175, TreeVec2{0.275, 0.725}),
        rectangle(TreeVec2{0.275, 0.45}, TreeVec2{0.55, 0.725}));
    return Glyph(0.55, _union(half, reflect_x(reflect_y(half, 0.5), 0.275)));
}

Glyph glyph_s(void) {
    const auto half = difference(
        circle(0.1625, TreeVec2{0.1625, 0.1625}),
        _union(
            scale_x(circle(0.0625, TreeVec2{0.165, 0.165}), 1.5, 0.165),
            rectangle(TreeVec2{0, 0.1625}, TreeVec2{0.1625, 0.325})));
    return Glyph(0.4875,
            scale_x(_union(half, reflect_x(reflect_y(half, 0.275), 0.1625)),
                    1.5, 0));
}

Glyph glyph_T(void) {
    return Glyph(0.6,
        _union(rectangle(TreeVec2{0, 0.9}, TreeVec2{0.6, 1}),
               rectangle(TreeVec2{0.25, 0}, TreeVec2{0.35, 1})));
}

Glyph glyph_t(void) {
    return Glyph(0.4,
        _union(intersection(ring(0.25, 0.15, TreeVec2{0.4, 0.25}),
                            rectangle(TreeVec2{0., 0}, TreeVec2{0.4, 0.25})),
        _union(rectangle(TreeVec2{0., 0.55}, TreeVec2{0.4, 0.65}),
               rectangle(TreeVec2{0.15, 0.25},  TreeVec2{0.25, 1}))));
}

Glyph glyph_U(void) {
    return Glyph(0.6,
        _union(_union(rectangle(TreeVec2{0, 0.3}, TreeVec2{0.1, 1}),
                      rectangle(TreeVec2{0.5, 0.3}, TreeVec2{0.6, 1})),
               difference(ring(0.3, 0.2, TreeVec2{0.3, 0.3}),
                          rectangle(TreeVec2{0, 0.3}, TreeVec2{0.6, 0.7}))));
}

Glyph glyph_u(void) {
    return Glyph(0.55,
        _union(intersection(ring(0.275, 0.175, TreeVec2{0.275, 0.275}),
                            rectangle(TreeVec2{0, 0}, TreeVec2{0.55, 0.275})),
        _union(rectangle(TreeVec2{0, 0.275}, TreeVec2{0.1, 0.55}),
               rectangle(TreeVec2{0.45, 0}, TreeVec2{0.55, 0.55}))));
}

Glyph glyph_V(void) {
    const auto half = _union(
        triangle(TreeVec2{0, 1}, TreeVec2{0.1, 1}, TreeVec2{0.35, 0}),
        triangle(TreeVec2{0.35, 0}, TreeVec2{0.25, 0}, TreeVec2{0, 1}));
    return Glyph(0.6, _union(half, reflect_x(half, 0.3)));
}

Glyph glyph_v(void) {
    const auto half = _union(
        triangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 0.55}, TreeVec2{0.35, 0}),
        triangle(TreeVec2{0.35, 0}, TreeVec2{0.25, 0}, TreeVec2{0, 0.55}));
    return Glyph(0.6, _union(half, reflect_x(half, 0.3)));
}

Glyph glyph_W(void) {
    const auto V = glyph_V();
    return Glyph(V.first * 2 - 0.1,
        _union(V.second, move(V.second, TreeVec3{V.first - 0.1, 0, 0})));
}

Glyph glyph_w(void) {
    const auto v = glyph_V();
    return Glyph(v.first * 2 - 0.1,
        _union(v.second, move(v.second, TreeVec3{v.first - 0.1, 0, 0})));
}

Glyph glyph_X(void) {
    const auto half = _union(
        triangle(TreeVec2{0, 1}, TreeVec2{0.125, 1}, TreeVec2{0.8, 0}),
        triangle(TreeVec2{0.8, 0}, TreeVec2{0.675, 0}, TreeVec2{0, 1}));
    return Glyph(0.8, _union(half, reflect_x(half, 0.4)));
}

Glyph glyph_x(void) {
    const auto half = _union(
        triangle(TreeVec2{0, 0.55}, TreeVec2{0.125, 0.55}, TreeVec2{0.55, 0}),
        triangle(TreeVec2{0.55, 0}, TreeVec2{0.425, 0}, TreeVec2{0, 0.55}));
    return Glyph(0.55, _union(half, reflect_x(half, 0.275)));
}
/*
YyZz ,.'":;!-)([]><°#1234567890+/

(make-glyph! #\Y 0.8
  union(difference(triangle(TreeVec2{0, 1}, TreeVec2{0.4, 0.5}, TreeVec2{0.8, 1})
                     triangle(TreeVec2{0.1, 1.01}, TreeVec2{0.4, 0.65}, TreeVec2{0.7, 1.01}))
         rectangle(TreeVec2{0.35, 0}, TreeVec2{0.45, 0.6})))

(make-glyph! #\y 0.55
 (let ((half union(triangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 0.55}, TreeVec2{0.325, 0})
                    triangle(TreeVec2{0.325, 0}, TreeVec2{0.225, 0}, TreeVec2{0, 0.55}))))
   union(half (reflect-x half 0.275)
          (move (reflect-x half 0.275) #[-0.225 -0.55]))))

(make-glyph! #\Z 0.6
  difference(rectangle(TreeVec2{0, 0}, TreeVec2{0.6, 1})
              triangle(TreeVec2{0, 0.1}, TreeVec2{0, 0.9}, TreeVec2{0.45, 0.9})
              triangle(TreeVec2{0.6, 0.1}, TreeVec2{0.15, 0.1}, TreeVec2{0.6, 0.9})))

(make-glyph! #\z 0.6
  difference(rectangle(TreeVec2{0, 0}, TreeVec2{0.6, 0.55})
              triangle(TreeVec2{0, 0.1}, TreeVec2{0, 0.45}, TreeVec2{0.45, 0.45})
              triangle(TreeVec2{0.6, 0.1}, TreeVec2{0.15, 0.1}, TreeVec2{0.6, 0.45})))

(make-glyph! #\space 0.55 (lambda-shape (x y z) 1))

(make-glyph! #\, 0.175
  union(circle(0.075, TreeVec2{0.1, 0.075})
    (intersection (scale-y circle(0.075, TreeVec2{0.075, 0.075}) 3 0.075)
                  rectangle(TreeVec2{0, -0.15}, TreeVec2{0.15, 0.075})
                  (inverse triangle(TreeVec2{0.1, 0.2}, TreeVec2{0, -0.15}, TreeVec2{-0.5, 0.075})))
))

(make-glyph! #\. 0.175 circle(0.075, TreeVec2{0.075, 0.075}))

(make-glyph! #\' 0.1 rectangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 0.8}))

(make-glyph! #\" 0.3
  union(rectangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 0.8})
         rectangle(TreeVec2{0.2, 0.55}, TreeVec2{0.3, 0.8})))

(make-glyph! #\: 0.15
  union(circle(0.075, TreeVec2{0.075, 0.15})
         circle(0.075, TreeVec2{0.075, 0.45})))

(make-glyph! #\; 0.15
  union(
    (move (glyph-shape #\,) #[0 0.075])
    (intersection (scale-y circle(0.074, TreeVec2{0.075, 0.15}) 3 0.015)
                  rectangle(TreeVec2{0, 0.15}, TreeVec2{-0.075, 0.15})
                  (inverse triangle(TreeVec2{0.075, 0.15}, TreeVec2{0, -0.075}, TreeVec2{-0.5, 0.15})))
    circle(0.075, TreeVec2{0.075, 0.45})))

(make-glyph! #\! 0.1
  union(rectangle(TreeVec2{0.025, 0.3}, TreeVec2{0.125, 1})
         circle(0.075, TreeVec2{0.075, 0.075})))

(make-glyph! #\- 0.45 rectangle(TreeVec2{0.05, 0.4}, TreeVec2{0.35, 0.5}))

(make-glyph! #\) 0.3
  (scale-x (intersection circle(0.6, TreeVec2{0, 0.5})
                         (inverse (scale-x circle(0.5, TreeVec2{0, 0.5}) 0.7))
                         rectangle(TreeVec2{0, 0}, TreeVec2{0.6, 1})) 0.5))

(make-glyph! #\( 0.3
  (reflect-x (glyph-shape #\)) (/ (glyph-width #\)) 2)))

(make-glyph! #\[ 0.3
  union(rectangle(TreeVec2{0, 0}, TreeVec2{0.1, 1})
         rectangle(TreeVec2{0, 0}, TreeVec2{0.3, 0.07})
         rectangle(TreeVec2{0, 0.93}, TreeVec2{0.3, 1})))

(make-glyph! #\] 0.3
  (reflect-x (glyph-shape #\[) (/ (glyph-width #\[) 2)))

(make-glyph! #\> 0.55
  (let ((half union(triangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 0.55}, TreeVec2{0.35, 0})
                     triangle(TreeVec2{0.35, 0}, TreeVec2{0.25, 0}, TreeVec2{0, 0.55}))))
    (move (rotate-z union(half (reflect-x half 0.3))
                    (/ pi 2))
          #[0.55 0.15] )))

(make-glyph! #\< 0.55
  (let ((half union(triangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 0.55}, TreeVec2{0.35, 0})
                     triangle(TreeVec2{0.35, 0}, TreeVec2{0.25, 0}, TreeVec2{0, 0.55}))))
     (move (rotate-z union(half (reflect-x half 0.3))
                     (* pi 1.5))
           #[0 0.75] )))

(make-glyph! #\° 0.4
  ring(0.175, 0.075, TreeVec2{0.2, 0.8}))

(make-glyph! #\# 0.55
  (move (shear-x-y  union(rectangle(TreeVec2{0.1, 0.05}, TreeVec2{0.2, 0.75})
                           rectangle(TreeVec2{0.3, 0.05}, TreeVec2{0.4, 0.75})
                           rectangle(TreeVec2{0, 0.25}, TreeVec2{0.5, 0.35})
                           rectangle(TreeVec2{0, 0.45}, TreeVec2{0.5, 0.55}))
                    TreeVec2{0, 0}, 0.35 0.1)
        TreeVec2{-0.05, 0}))

(make-glyph! #\1 0.3
  difference(rectangle(TreeVec2{0, 0}, TreeVec2{0.3, 1})
              rectangle(TreeVec2{0, 0}, TreeVec2{0.2, 0.75})
              circle(0.2, #[0 1])))

(make-glyph! #\2 0.55
  union(difference(ring(0.275, 0.175, TreeVec2{0.275, 0.725})
                     rectangle(TreeVec2{0, 0}, TreeVec2{0.55, 0.725}))
         rectangle(TreeVec2{0, 0}, TreeVec2{0.55, 0.1})
         triangle(TreeVec2{0, 0.1}, TreeVec2{0.45, 0.775}, TreeVec2{0.55, 0.725})
         triangle(TreeVec2{0, 0.1}, TreeVec2{0.55, 0.725}, TreeVec2{0.125, 0.1})))

(make-glyph! #\3 0.55
  (difference
    union(ring(0.275, 0.175, TreeVec2{0.3, 0.725})
           ring(0.275, 0.175, TreeVec2{0.3, 0.225}))
    rectangle(TreeVec2{0, 0.275}, TreeVec2{0.275, 0.725})))

(make-glyph! #\4 0.5
  (intersection union(triangle(TreeVec2{-0.10, 0.45}, TreeVec2{0.4, 1}, TreeVec2{0.4, 0.45})
                       rectangle(TreeVec2{0.4, 0}, TreeVec2{0.5, 1}))
              (inverse triangle(TreeVec2{0.4, 0.85}, TreeVec2{0.4, 0.55}, TreeVec2{0.1, 0.55}))
              rectangle(TreeVec2{0, 0}, TreeVec2{0.5, 1})))

(make-glyph! #\5 0.65
  union(difference(ring(0.325, 0.225, TreeVec2{0.325, 0.325})
                     rectangle(TreeVec2{0, 0.325}, TreeVec2{0.325, 0.65}))
         rectangle(TreeVec2{0, 0.55}, TreeVec2{0.325, 0.65})
         rectangle(TreeVec2{0, 0.55}, TreeVec2{0.1, 1})
         rectangle(TreeVec2{0.1, 0.9}, TreeVec2{0.65, 1})))

(make-glyph! #\6 0.55
  (let ((hook
    (intersection
      circle(0.275, TreeVec2{0.275, 0.725})
      (inverse (scale-y circle(0.175, TreeVec2{0.275, 0.725}) 1.2 0.725))
      rectangle(TreeVec2{0, 0.725}, TreeVec2{0.55, 1})
      (inverse triangle(TreeVec2{0.275, 0.925}, TreeVec2{0.55, 0.9}, TreeVec2{0.55, 0.525})))))
  union(ring(0.275, 0.175, TreeVec2{0.275, 0.275})
         rectangle(TreeVec2{0, 0.275}, TreeVec2{0.1, 0.45})
         difference((scale-x (scale-y hook 2 1) 1.1)
                     rectangle(TreeVec2{0.275, 0.65}, TreeVec2{0, 0.7})))))

(make-glyph! #\7 0.6
  union(rectangle(TreeVec2{0, 0.9}, TreeVec2{0.6, 1})
         (triangle #[0 0] #[0.475 0.9] #[0.6 0.9])
         (triangle #[0 0] #[0.6 0.9] #[0.125 0])))

(make-glyph! #\8 0.55
  union(ring(0.275, 0.175, TreeVec2{0.3, 0.725})
         ring(0.275, 0.175, TreeVec2{0.3, 0.225})))

(make-glyph! #\9 (glyph-width #\6)
  (reflect-x (reflect-y (glyph-shape #\6) 0.5)
             (/ (glyph-width #\6) 2)))

(make-glyph! #\0 0.7
  (scale-x difference(circle(0.5, TreeVec2{0.5, 0.5})
                       (scale-x circle(0.4, TreeVec2{0.5, 0.5}) (sqrt 0.7) 0.5))
           0.7))

(make-glyph! #\+ 0.55
  union(rectangle(TreeVec2{0, 0.45}, TreeVec2{0.5, 0.55})
         rectangle(TreeVec2{0.2, 0.25}, TreeVec2{0.3, 0.75})))

(make-glyph! #\/ 0.55
  union(triangle(TreeVec2{0, 0}, TreeVec2{0.425, 1}, TreeVec2{0.55, 1})
         triangle(TreeVec2{0, 0}, TreeVec2{0.55, 1}, TreeVec2{0.125, 0})))

(make-glyph! #\? 0.55
  union(difference(ring(0.275, 0.175, TreeVec2{0.275, 0.725})
                     rectangle(TreeVec2{0, 0.45}, TreeVec2{0.275, 0.725}))
         rectangle(TreeVec2{0.225, 0.3}, TreeVec2{0.325, 0.55})
         circle(0.075, TreeVec2{0.275, 0.075})))
*/
