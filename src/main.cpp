// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021-2023 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#define _USE_MATH_DEFINES

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"

// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline
{
public:
  explicit CubicSpline(const Real h = 1) : _dim(2)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real h)
  {
    const Real h2 = square(h), h3 = h2 * h;
    _h = h;
    _sr = 2e0 * h;
    _c[0] = 2e0 / (3e0 * h);
    _c[1] = 10e0 / (7e0 * M_PI * h2);
    _c[2] = 1e0 / (M_PI * h3);
    _gc[0] = _c[0] / h;
    _gc[1] = _c[1] / h;
    _gc[2] = _c[2] / h;
  }
  Real smoothingLen() const { return _h; }
  Real supportRadius() const { return _sr; }

  Real f(const Real l) const
  {
    const Real q = l / _h;
    if (q < 1e0)
      return _c[_dim - 1] * (1e0 - 1.5 * square(q) + 0.75 * cube(q));
    else if (q < 2e0)
      return _c[_dim - 1] * (0.25 * cube(2e0 - q));
    return 0;
  }
  Real derivative_f(const Real l) const
  {
    const Real q = l / _h;
    if (q <= 1e0)
      return _gc[_dim - 1] * (-3e0 * q + 2.25 * square(q));
    else if (q < 2e0)
      return -_gc[_dim - 1] * 0.75 * square(2e0 - q);
    return 0;
  }

  Real w(const Vec2f &rij) const { return f(rij.length()); }
  Vec2f grad_w(const Vec2f &rij) const { return grad_w(rij, rij.length()); }
  Vec2f grad_w(const Vec2f &rij, const Real len) const
  {
    return derivative_f(len) * rij / len;
  }

private:
  unsigned int _dim;
  Real _h, _sr, _c[3], _gc[3];
};

class SphSolver
{
public:
  explicit SphSolver(
      const Real nu = 0.08, const Real epsilon = 0.05f, const Real h = 0.5, const Real density = 1e3,
      const Vec2f g = Vec2f(0, -9.8), const Real eta = 0.01, const Real gamma = 7.0) : _kernel(h), _nu(nu), _epsilon(epsilon), _h(h), _d0(density),
                                                                                       _g(g), _eta(eta), _gamma(gamma)
  {
    _dt = 0.0005;
    _m0 = _d0 * _h * _h;
    _c = std::fabs(_g.y) / _eta;
    _k = _d0 * _c * _c / _gamma;
  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
      const int res_x, const int res_y, const int f_width, const int f_height)
  {
    _pos.clear();

    _resX = res_x;
    _resY = res_y;

    _sample = 2; // only works for 2. density pushes too far particles otherwise, they come out of the boundaries
    const int sample_i = _sample;
    const int sample_j = _sample;
    float kernel_radius = _kernel.supportRadius();

    // sample solid walls
    float x_min = 1.0f / (2 * sample_i);
    float y_min = 1.0f / (2 * sample_j);
    float x_max = res_x - x_min;
    float y_max = res_y - y_min;
    int wall_width = ceil(kernel_radius);
    for (size_t i = 0; i < res_x; i++)
    {
      for (size_t w = 0; w < wall_width; w++)
      {
        for (size_t sub_i = 0; sub_i < sample_i; sub_i++)
        {
          for (size_t sub_j = 0; sub_j < sample_j; sub_j++)
          {
            float x = i + (1.0f + 2 * float(sub_i)) / (2 * sample_i);
            float y = y_min + float(w) + float(sub_j) / sample_j;
            _pos.push_back(Vec2f(x, y));
            _n.push_back(Vec2f(0.0f, 1.0f));
            y = y_max - float(w) - float(sub_j) / sample_j;
            _pos.push_back(Vec2f(x, y));
            _n.push_back(Vec2f(0.0f, -1.0f));
          }
        }
      }
    }
    for (size_t j = wall_width; j < res_y - wall_width; j++)
    {
      for (size_t w = 0; w < wall_width; w++)
      {
        for (size_t sub_j = 0; sub_j < sample_i; sub_j++)
        {
          for (size_t sub_i = 0; sub_i < sample_j; sub_i++)
          {
            float x = x_min + float(w) + float(sub_i) / sample_i;
            float y = j + (1.0f + 2 * float(sub_j)) / (2 * sample_j);
            _pos.push_back(Vec2f(x, y));
            _n.push_back(Vec2f(0.0f, 1.0f));
            x = x_max - float(w) - float(sub_i) / sample_i;
            _pos.push_back(Vec2f(x, y));
            _n.push_back(Vec2f(0.0f, -1.0f));
          }
        }
      }
    }

    // sample a solid mass
    int center_x = res_x / 2;
    int center_y = 3 * res_y / 5;
    Vec2f center = Vec2f(center_x, center_y);
    float radius = fmin(f_width, f_height) * 2;
    for (int i = center_x - radius; i < center_x + radius; i++)
    {
      for (int j = center_y - radius; j < center_y + radius; j++)
      {
        for (size_t sub_i = 0; sub_i < sample_i; sub_i++)
        {
          for (size_t sub_j = 0; sub_j < sample_j; sub_j++)
          {
            float x = i + (1.0f + 2 * float(sub_i)) / (2 * sample_i);
            float y = j + (1.0f + 2 * float(sub_j)) / (2 * sample_j);
            const Vec2f pos = Vec2f(x, y);
            const Vec2f relPos = pos - center;
            float dist_2 = relPos.lengthSquare();
            if (dist_2 < pow(radius, 2) && dist_2 > pow(radius - kernel_radius, 2))
            {
              _pos.push_back(pos);
              const Vec2f n = relPos.normalized();
              _n.push_back(n);
            }
          }
        }
      }
    }

    _nS = _pos.size();

    // sample a fluid mass
    center_x = res_x / 2;
    center_y = 4 * res_y / 5;
    radius = fmin(f_width, f_height);
    for (int i = center_x - radius; i < center_x + radius; i++)
    {
      for (int j = center_y - radius; j < center_y + radius; j++)
      {
        for (size_t sub_i = 0; sub_i < sample_i; sub_i++)
        {
          for (size_t sub_j = 0; sub_j < sample_j; sub_j++)
          {
            float x = i + (1.0f + 2 * float(sub_i)) / (2 * sample_i);
            float y = j + (1.0f + 2 * float(sub_j)) / (2 * sample_j);
            float dist_2 = pow(float(x - center_x), 2) + pow(float(y - center_y), 2);
            // if (dist_2 < pow(radius, 2))
            {
              _pos.push_back(Vec2f(x, y));
            }
          }
        }
      }
    }

    _nL = _pos.size() - _nS;

    // sample perfect ghost air
    radius += kernel_radius;
    for (int i = center_x - radius; i < center_x + radius; i++)
    {
      for (int j = center_y - radius; j < center_y + radius; j++)
      {
        for (size_t sub_i = 0; sub_i < sample_i; sub_i++)
        {
          for (size_t sub_j = 0; sub_j < sample_j; sub_j++)
          {
            float x = i + (1.0f + 2 * float(sub_i)) / (2 * sample_i);
            float y = j + (1.0f + 2 * float(sub_j)) / (2 * sample_j);
            float dist_2 = pow(float(x - center_x), 2) + pow(float(y - center_y), 2);
            // if (dist_2 < pow(radius, 2) && dist_2 > pow(radius - kernel_radius, 2))
            if (abs(x - center_x) > abs(radius - kernel_radius) || abs(y - center_y) > abs(radius - kernel_radius))
            {
              _pos.push_back(Vec2f(x, y));
            }
          }
        }
      }
    }

    _nA = _pos.size() - _nS - _nL;
    _nP = _pos.size();
    _closestLiquidParts = std::vector<std::pair<tIndex, Vec2f>>(_nP, {-1, Vec2f(_kernel.supportRadius())});
    _d = std::vector<Real>(_nP, 0);
    for (size_t partId = _nS + _nL; partId < _nP; partId++)
    {
      _d[partId] = _d0;
    }
    _p = std::vector<Real>(_nP, 0);
    _vel = std::vector<Vec2f>(_nP, Vec2f(0, 0));
    _acc = std::vector<Vec2f>(_nP, Vec2f(0, 0));
    _velStar = std::vector<Vec2f>(_nP, Vec2f(0, 0));
    _col = std::vector<float>(_nP * 4, 1.0); // RGBA
    _vln = std::vector<float>(_nP * 4, 0.0); // GL_LINES

    updateColor();

    sampleNewAir();
  }

  void sampleNewAir()
  {
    // For air
    _pos.resize(_nS + _nL);
    const float minR = 1.0f / _sample; // * sqrt(2.0f); // minimum distance
    const float minR2 = minR * minR;
    const int k = 30; // limit of samples
    std::vector<Vec2f> list;
    for (size_t i = 0; i < _nS + _nL; i++)
    {
      list.push_back(_pos[i]);
    }
    std::vector<Vec2f> activeList;
    for (size_t i = _nS; i < _nS + _nL; i++)
    {
      activeList.push_back(_pos[i]);
    }
    int range = activeList.size();
    // for (size_t i = 0; i < 10; i++)
    while (0 < range)
    {
      const int seed = range - 1; // rand() % range;
      const Vec2f center = activeList[seed];
      for (size_t i = 0; i < k; i++)
      {
        bool add = true;
        const float r2 = minR2 + (_kernel.supportRadius() - minR) * (float)(rand()) / (float)(RAND_MAX);
        const float r = sqrt(r2);
        const float theta = 2 * M_PI * (float)(rand()) / (float)(RAND_MAX);
        const Vec2f _sample = center + Vec2f(r * cos(theta), r * sin(theta));
        for (const Vec2f &part : list)
        {
          if ((part - _sample).lengthSquare() < minR2)
          {
            // std::cout << "false" << std::endl;
            add = false;
            break;
          }
        }
        if (add)
        {
          list.push_back(_sample);
          _pos.push_back(_sample);
        }
      }
      // if (!add)
      // {
      activeList.pop_back();
      // }

      range = activeList.size();
    }
    _nP = _pos.size();
    _nA = _nP - _nS - _nL;
    _closestLiquidParts.resize(_nS + _nL);
    _d.resize(_nS + _nL);
    _p.resize(_nS + _nL);
    _vel.resize(_nS + _nL);
    _acc.resize(_nS + _nL);
    _velStar.resize(_nS + _nL);
    _col.resize(4 * (_nS + _nL));
    _vln.resize(4 * (_nS + _nL));
    for (size_t i = 0; i < _nA; i++)
    {
      _closestLiquidParts.push_back({-1, Vec2f(_kernel.supportRadius())});
      _d.push_back(_d0);
      _p.push_back(0.0f);
      _vel.push_back(Vec2f());
      _acc.push_back(Vec2f());
      _velStar.push_back(Vec2f());
      for (size_t j = 0; j < 4; j++)
      {
        _col.push_back(1.0f);
        _vln.push_back(0.0f);
      }
    }
    updateColor();
    if (gShowVel)
      updateVelLine();
  }

  void update()
  {
    std::cout << '.' << std::flush;

    buildNeighbor();
    computeDensity();
    computePressure();

    _acc = std::vector<Vec2f>(_nP, Vec2f(0, 0));
    applyBodyForce();
    applyPressureForce();

    updateVelocityStar();

    applyViscousForce();

    updateVelocity();
    updatePosition();

    updateColor();
    if (gShowVel)
      updateVelLine();
  }

  tIndex particleCount() const { return _nP; }
  const Vec2f &position(const tIndex i) const { return _pos[i]; }
  const float &color(const tIndex i) const { return _col[i]; }
  const float &vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }

  Real equationOfState(
      const Real d, const Real d0,
      const bool liquid = true,
      const Real k = 20000, // NOTE: You can use _k for k here.
      const Real gamma = 7.0)
  {
    if (liquid)
    {
      return k * (pow(d / d0, gamma) - 1.0f);
    }
    else
    {
      const float value = k * (pow(d / d0, gamma) - 1.0f);
      if (0 < value)
      {
        return 10 * value;
      }
      else
      {
        return 10 * value;
      }
    }
  }

private:
  void buildNeighbor()
  {
    // For all
    _pidxInGrid.clear();
    _pidxInGrid.resize(_resX * _resY);
    for (size_t partId = 0; partId < _nP; partId++)
    {
      const Vec2f pos = _pos[partId];
      const int iCell = pos[0];
      const int jCell = pos[1];
      const int cellId = idx1d(iCell, jCell);
      _pidxInGrid[cellId].push_back(partId);
    }
    _supportParts.clear();
    _supportParts.resize(_nP);
    const int kernelReach = ceil(_kernel.supportRadius());
    for (size_t partId = 0; partId < _nP; partId++)
    {
      const Vec2f pos = _pos[partId];
      const int iCentralCell = pos[0];
      const int jCentralCell = pos[1];
      std::vector<tIndex> reachedCellsIds;
      const int lowerBoundX = iCentralCell - fmin(kernelReach, iCentralCell);
      const int upperBoundX = iCentralCell + fmin(kernelReach, _resX - 1 - iCentralCell) + 1;
      const int lowerBoundY = jCentralCell - fmin(kernelReach, jCentralCell);
      const int upperBoundY = jCentralCell + fmin(kernelReach, _resY - 1 - jCentralCell) + 1;
      for (size_t iCell = lowerBoundX; iCell < upperBoundX; iCell++)
      {
        for (size_t jCell = lowerBoundY; jCell < upperBoundY; jCell++)
        {
          for (auto neighborPartId : _pidxInGrid[idx1d(iCell, jCell)])
          {
            const Vec2f consideredPos = _pos[neighborPartId];
            const Vec2f rij = pos - consideredPos;
            if (rij.lengthSquare() <= pow(_kernel.supportRadius(), 2))
            {
              _supportParts[partId].push_back({neighborPartId, rij});
            }
          }
        }
      }
    }

    // For solid
    for (size_t partId = 0; partId < _nS; partId++)
    {
      std::pair<tIndex, Vec2f> closestLiquidPart = {-1, Vec2f(_kernel.supportRadius())};
      for (auto influencingPart : _supportParts[partId])
      {
        const tIndex neighborId = influencingPart.first;
        if (_nS <= neighborId && neighborId < _nS + _nL) // the particle is liquid
        {
          const Vec2f rij = influencingPart.second;
          if (rij.lengthSquare() < closestLiquidPart.second.lengthSquare())
          {
            closestLiquidPart = influencingPart;
          }
        }
      }
      _closestLiquidParts[partId] = closestLiquidPart;
    }
    // For air
    for (size_t partId = _nS + _nL; partId < _nP; partId++)
    {
      std::pair<tIndex, Vec2f> closestLiquidPart = {-1, Vec2f(_kernel.supportRadius())};
      for (auto influencingPart : _supportParts[partId])
      {
        const tIndex neighborId = influencingPart.first;
        // std::cout << "air neighborId: " << neighborId << std::endl;
        if (_nS <= neighborId && neighborId < _nS + _nL) // the particle is liquid
        {
          const Vec2f rij = influencingPart.second;
          if (rij.lengthSquare() < closestLiquidPart.second.lengthSquare())
          {
            closestLiquidPart = influencingPart;
          }
        }
      }
      // std::cout << "chosen as closest" << closestLiquidPart.first << std::endl;
      _closestLiquidParts[partId] = closestLiquidPart;
    }
  }

  void computeDensity()
  {
    // For liquid:
    for (size_t partId = _nS; partId < _nS + _nL; partId++)
    {
      Real inverseVolume = 0;
      for (auto influencingPart : _supportParts[partId])
      {
        const Vec2f rij = influencingPart.second;
        inverseVolume += _kernel.w(rij);
      }
      _d[partId] = _m0 * inverseVolume;
    }

    // For solid
    for (size_t partId = 0; partId < _nS; partId++)
    {
      std::pair<tIndex, Vec2f> closestLiquidPart = _closestLiquidParts[partId];
      const tIndex closestPartId = closestLiquidPart.first;
      if (closestPartId == -1)
      {
        _d[partId] = _d0;
      }
      else
      {
        _d[partId] = _d[closestPartId];
      }
    }
  }

  void computePressure()
  {
    // For solid
    for (size_t partId = 0; partId < _nS; partId++)
    {
      _p[partId] = equationOfState(_d[partId], _d0, false);
    }
    // For liquid and air
    for (size_t partId = _nS; partId < _nP; partId++)
    {
      _p[partId] = equationOfState(_d[partId], _d0);
    }
  }

  void applyBodyForce()
  {
    // For liquid:
    for (size_t partId = _nS; partId < _nS + _nL; partId++)
    {
      _acc[partId] += _g;
    }
  }

  void applyPressureForce()
  {
    // For liquid
    for (size_t partId = _nS; partId < _nS + _nL; partId++)
    {
      Vec2f pressureForceCoeff;
      for (auto part : _supportParts[partId])
      {
        const tIndex neighborPartId = part.first;
        if (neighborPartId != partId && neighborPartId < _nS + _nL)
        {
          const Vec2f rij = part.second;
          pressureForceCoeff += (_p[partId] / pow(_d[partId], 2) + _p[neighborPartId] / pow(_d[neighborPartId], 2)) * _kernel.grad_w(rij);
        }
      }
      _acc[partId] -= _m0 * pressureForceCoeff;
    }
  }

  void updateVelocityStar()
  {
    // For liquid
    for (size_t partId = _nS; partId < _nS + _nL; partId++)
    {
      _velStar[partId] = _vel[partId] + _dt * _acc[partId];
    }

    // For solid
    for (size_t partId = 0; partId < _nS; partId++)
    {
      const Vec2f n = _n[partId];
      const Vec2f viN = _vel[partId].dotProduct(n) * n;
      _velStar[partId] = viN;
      const std::pair<tIndex, Vec2f> closestLiquidPart = _closestLiquidParts[partId];
      const tIndex closestLiquidPartId = closestLiquidPart.first;
      if (closestLiquidPartId != -1)
      {
        const Vec2f vjT = _vel[closestLiquidPartId] - _vel[closestLiquidPartId].dotProduct(n) * n;
        _velStar[partId] += vjT;
      }
    }
  }

  void oldApplyViscousForce()
  {
    // For liquid
    for (size_t partId = _nS; partId < _nS + _nL; partId++)
    {
      Vec2f viscosityForceCoeff;
      for (auto part : _supportParts[partId])
      {
        const tIndex neighborPartId = part.first;
        if (neighborPartId != partId)
        {
          const Vec2f rij = part.second;
          const Vec2f uij = _vel[partId] - _vel[neighborPartId];
          viscosityForceCoeff += uij / _d[neighborPartId] * rij.dotProduct(_kernel.grad_w(rij)) / (rij.dotProduct(rij) + 0.01f * _h * _h);
        }
      }
      _acc[partId] += 2 * _nu * _m0 * viscosityForceCoeff;
    }
  }

  void applyViscousForce()
  {
    // For liquid
    for (size_t partId = _nS; partId < _nS + _nL; partId++)
    {
      Vec2f viscosityForceCoeff;
      for (auto part : _supportParts[partId])
      {
        const tIndex neighborPartId = part.first;
        if (neighborPartId != partId && neighborPartId < _nS + _nL)
        {
          const Vec2f rij = part.second;
          const Vec2f uijStar = _velStar[neighborPartId] - _velStar[partId];
          viscosityForceCoeff += uijStar / _d[neighborPartId] * _kernel.w(rij);
        }
      }
      _vel[partId] = _velStar[partId] + _epsilon * _m0 * viscosityForceCoeff;
    }
  }

  void updateVelocity()
  {
    // For air
    for (size_t partId = _nS + _nL; partId < _nP; partId++)
    {
      std::pair<tIndex, Vec2f> closestLiquidPart = _closestLiquidParts[partId];
      const tIndex closestLiquidPartId = closestLiquidPart.first;
      if (closestLiquidPartId == -1)
      {
        _vel[partId] = Vec2f();
      }
      else
      {
        _vel[partId] = _vel[closestLiquidPart.first];
      }
    }
  }

  void updatePosition()
  {
    // For liquid and air
    for (size_t partId = _nS; partId < _nP; partId++)
    {
      _pos[partId] += _dt * _vel[partId];
    }
  }

  void updateColor()
  {
    for (tIndex i = 0; i < _nS; ++i)
    {
      _col[i * 4 + 0] = 1.0f;
      _col[i * 4 + 1] = 0.5f;
      _col[i * 4 + 2] = 0.0f;
    }
    for (tIndex i = _nS; i < _nS + _nL; ++i)
    {
      _col[i * 4 + 0] = 0.0f;
      _col[i * 4 + 1] = 0.6f;
      _col[i * 4 + 2] = 1.0f;
    }
    for (tIndex i = _nS + _nL; i < particleCount(); ++i)
    {
      _col[i * 4 + 0] = 1.0f;
      _col[i * 4 + 1] = 1.0f;
      _col[i * 4 + 2] = 1.0f;
    }
  }

  void updateVelLine()
  {
    for (tIndex i = _nS; i < particleCount(); ++i)
    {
      _vln[i * 4 + 0] = _pos[i].x;
      _vln[i * 4 + 1] = _pos[i].y;
      _vln[i * 4 + 2] = _pos[i].x + _vel[i].x;
      _vln[i * 4 + 3] = _pos[i].y + _vel[i].y;
    }
  }

  inline tIndex idx1d(const int i, const int j) { return i + j * resX(); }

  const CubicSpline _kernel;

  // particle data
  std::vector<Vec2f> _pos;     // position
  std::vector<Vec2f> _vel;     // velocity
  std::vector<Vec2f> _velStar; // advanced/ghost velocity
  std::vector<Vec2f> _acc;     // acceleration
  std::vector<Real> _p;        // pressure
  std::vector<Real> _d;        // density
  std::vector<Vec2f> _n;       // normal

  int _nS; // number of solid particles
  int _nL; // number of liquid particles
  int _nA; // number of air particles
  int _nP; // number of particles

  std::vector<std::vector<tIndex>> _pidxInGrid; // will help you find neighbor particles
  std::vector<std::vector<std::pair<tIndex, Vec2f>>> _supportParts;
  std::vector<std::pair<tIndex, Vec2f>> _closestLiquidParts;

  std::vector<float> _col; // particle color; just for visualization
  std::vector<float> _vln; // particle velocity lines; just for visualization

  // simulation
  Real _dt; // time step

  int _resX, _resY; // background grid resolution
  int _sample;      // nb of particles per segment of a grid

  // SPH coefficients
  Real _nu; // viscosity coefficient
  Real _epsilon;
  Real _d0; // rest density
  Real _h;  // particle spacing (i.e., diameter)
  Vec2f _g; // gravity

  Real _m0; // rest mass
  Real _k;  // EOS coefficient

  Real _eta;
  Real _c;     // speed of sound
  Real _gamma; // EOS power factor
};

SphSolver gSolver(0.08, 0.05f, 0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp()
{
  std::cout << "> Help:" << std::endl
            << "    Keyboard commands:" << std::endl
            << "    * H: print this help" << std::endl
            << "    * P: toggle simulation" << std::endl
            << "    * G: toggle grid rendering" << std::endl
            << "    * V: toggle velocity rendering" << std::endl
            << "    * S: save current frame into a file" << std::endl
            << "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow *window, int width, int height)
{
  gWindowWidth = width;
  gWindowHeight = height;
  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if (action == GLFW_PRESS && key == GLFW_KEY_H)
  {
    printHelp();
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_S)
  {
    gSaveFile = !gSaveFile;
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_G)
  {
    gShowGrid = !gShowGrid;
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_V)
  {
    gShowVel = !gShowVel;
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_P)
  {
    gAppTimerStoppedP = !gAppTimerStoppedP;
    if (!gAppTimerStoppedP)
      gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
  }
  else if (action == GLFW_PRESS && key == GLFW_KEY_Q)
  {
    glfwSetWindowShouldClose(window, true);
  }
}

void initGLFW()
{
  // Initialize GLFW, the library responsible for window management
  if (!glfwInit())
  {
    std::cerr << "ERROR: Failed to init GLFW" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Before creating the window, set some option flags
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create the window
  gWindowWidth = gSolver.resX() * kViewScale;
  gWindowHeight = gSolver.resY() * kViewScale;
  gWindow = glfwCreateWindow(
      gSolver.resX() * kViewScale, gSolver.resY() * kViewScale,
      "Basic SPH Simulator", nullptr, nullptr);
  if (!gWindow)
  {
    std::cerr << "ERROR: Failed to open window" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  // Load the OpenGL context in the GLFW window
  glfwMakeContextCurrent(gWindow);

  // not mandatory for all, but MacOS X
  glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

  // Connect the callbacks for interactive control
  glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
  glfwSetKeyCallback(gWindow, keyCallback);

  std::cout << "Window created: " << gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string &message)
{
  std::cerr << "> [Critical error]" << message << std::endl;
  std::cerr << "> [Clearing resources]" << std::endl;
  clear();
  std::cerr << "> [Exit]" << std::endl;
  std::exit(EXIT_FAILURE);
}

void initOpenGL()
{
  // Load extensions for modern OpenGL
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    exitOnCriticalError("[Failed to initialize OpenGL context]");

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
  gSolver.initScene(48, 64, 4, 4);

  initGLFW(); // Windowing system
  initOpenGL();
}

void clear()
{
  glfwDestroyWindow(gWindow);
  glfwTerminate();
}

// The main rendering call
void render()
{
  const float backColor = 0.3f;
  glClearColor(backColor, backColor, backColor, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // grid guides
  if (gShowGrid)
  {
    const float gridCol = backColor + 0.05f;
    glBegin(GL_LINES);
    for (int i = 1; i < gSolver.resX(); ++i)
    {
      glColor3f(gridCol, gridCol, gridCol);
      glVertex2f(static_cast<Real>(i), 0.0);
      glColor3f(gridCol, gridCol, gridCol);
      glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
    }
    for (int j = 1; j < gSolver.resY(); ++j)
    {
      glColor3f(gridCol, gridCol, gridCol);
      glVertex2f(0.0, static_cast<Real>(j));
      glColor3f(gridCol, gridCol, gridCol);
      glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
    }
    glEnd();
  }

  // render particles
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(0.65f * kViewScale);

  glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
  glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
  glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // velocity
  if (gShowVel)
  {
    glColor4f(0.5f, 0.6f, 0.0f, 0.2f);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
    glDrawArrays(GL_LINES, 0, gSolver.particleCount() * 2);

    glDisableClientState(GL_VERTEX_ARRAY);
  }

  if (gSaveFile)
  {
    std::stringstream fpath;
    fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";

    std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
    const short int w = gWindowWidth;
    const short int h = gWindowHeight;
    std::vector<int> buf(w * h * 3, 0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

    FILE *out = fopen(fpath.str().c_str(), "wb");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(&(buf[0]), 3 * w * h, 1, out);
    fclose(out);
    gSaveFile = false;

    std::cout << "Done" << std::endl;
  }
}

int STEP = 0;

// Update any accessible variable based on the current time
void update(const float currentTime)
{
  if (!gAppTimerStoppedP)
  {
    std::cout << "STEP: " << STEP << std::endl;
    STEP++;
    // NOTE: When you want to use application's dt ...
    const float dt = currentTime - gAppTimerLastClockTime;
    gAppTimerLastClockTime = currentTime;
    gAppTimer += dt;

    if (STEP % 10 == 0)
    {
      gSolver.sampleNewAir();
    }

    // solve 10 steps
    for (int i = 0; i < 10; ++i)
    {
      gSolver.update();
    }
  }
}

int main(int argc, char **argv)
{
  init();
  while (!glfwWindowShouldClose(gWindow))
  {
    update(static_cast<float>(glfwGetTime()));
    render();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
  }
  clear();
  std::cout << " > Quit" << std::endl;
  return EXIT_SUCCESS;
}
