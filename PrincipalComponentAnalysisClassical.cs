using System;
using System.IO;

// PCA classical using Jacobi eigenvectors

namespace PrincipalComponentsClassic
{
  internal class PrincipalClassicProgram
  {
    public static void MainPCA(string[] args)
    {
      Console.WriteLine("\nBegin classic PCA (Jacobi) " +
        "from scratch using C# ");

      // 0. load data
      Console.WriteLine("\nLoading Iris-9 data ");
      string dataFile = "..\\..\\..\\Data\\iris_9.txt";
      double[][] X = MatLoad(dataFile, new int[] { 0, 1, 2, 3 },
        ',', "#");
      Console.WriteLine("Done ");

      Console.WriteLine("\nSource data: ");
      MatShow(X, 4, 9, true);

      // 1. standardize data
      Console.WriteLine("\nStandardizing (z-score," +
        " biased) ");
      double[] means;
      double[] stds;
      double[][] stdX = MatStandardize(X,
        out means, out stds);

      Console.WriteLine("\nStandardized data items ");
      MatShow(stdX, 4, 9, nRows: 3);

      // 2. covariance matrix of standardized data
      Console.WriteLine("\nComputing covariance matrix: ");
      double[][] covarMat = CovarMatrix(stdX, false);
      MatShow(covarMat, 4, 9, false);  // allow sym Jacobi

      // 3. eigenvalues, eigenvectors Jacobi algorithm
      Console.WriteLine("\nComputing eigenvalues and" +
        " eigenvectors ");
      double[] eigenVals;
      double[][] eigenVecs;
      Eigen(covarMat, out eigenVals, out eigenVecs);
      Console.WriteLine("\nEigenvectors (not sorted," +
        " as cols): ");
      MatShow(eigenVecs, 4, 9, false);

      Console.WriteLine("\nEigenvalues (not sorted):");
      VecShow(eigenVals, 4, 9);

      // 4. sort eigenvals from large to smallest
      int[] idxs = ArgSort(eigenVals);  // to sort evecs
      Array.Reverse(idxs);
      Array.Sort(eigenVals);
      Array.Reverse(eigenVals);
      Console.WriteLine("\nEigenvalues (sorted): ");
      VecShow(eigenVals, 4, 9);

      // 5. sort associated eigenvectors
      eigenVecs = MatExtractCols(eigenVecs, idxs); // sort
      eigenVecs = MatTranspose(eigenVecs);  // as rows
      Console.WriteLine("\nEigenvectors (sorted as rows):");
      MatShow(eigenVecs, 4, 9, false);

      // 6. variance explained to guide number components
      Console.WriteLine("\nComputing variance" +
        " explained: ");
      double[] varExplained = VarExplained(eigenVals);
      VecShow(varExplained, 4, 9);

      // 7. transform all data
      Console.WriteLine("\nComputing transformed" +
        " data (all components): ");
      double[][] transformed =
        MatProduct(stdX, MatTranspose(eigenVecs));  // all 
      Console.WriteLine("\nTransformed data: ");
      MatShow(transformed, 4, 9, true);

      // 8. reconstruct data just for fun / to verify
      Console.WriteLine("\nReconstructing " +
       " original data: ");
      double[][] reconstructed =
        MatProduct(transformed, eigenVecs);
      for (int i = 0; i < reconstructed.Length; ++i)
        for (int j = 0; j < reconstructed[0].Length; ++j)
          reconstructed[i][j] =
            (reconstructed[i][j] * stds[j]) + means[j];
      MatShow(reconstructed, 4, 9, 3);

      Console.WriteLine("\nEnd demo ");
      Console.ReadLine();

    } // Main

    // ------------------------------------------------------

    static double[][] MatLoad(string fn, int[] usecols,
      char sep, string comment)
    {
      // completely self-contained version
      // count number of non-comment lines
      int nRows = 0;
      string line = "";
      FileStream ifs = new FileStream(fn, FileMode.Open);
      StreamReader sr = new StreamReader(ifs);
      while ((line = sr.ReadLine()) != null)
        if (line.StartsWith(comment) == false)
          ++nRows;
      sr.Close(); ifs.Close();

      int nCols = usecols.Length;
      double[][] result = new double[nRows][];
      for (int r = 0; r < nRows; ++r)
        result[r] = new double[nCols];

      line = "";
      string[] tokens = null;
      ifs = new FileStream(fn, FileMode.Open);
      sr = new StreamReader(ifs);

      int i = 0;
      while ((line = sr.ReadLine()) != null)
      {
        if (line.StartsWith(comment) == true)
          continue;
        tokens = line.Split(sep);
        for (int j = 0; j < nCols; ++j)
        {
          int k = usecols[j];  // into tokens
          result[i][j] = double.Parse(tokens[k]);
        }
        ++i;
      }
      sr.Close(); ifs.Close();
      return result;
    }

    // ------------------------------------------------------

    static void MatShow(double[][] M, int dec,
      int wid, bool showIndices)
    {
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < M.Length; ++i)
      {
        if (showIndices == true)
        {
          int pad = M.Length.ToString().Length;
          Console.Write("[" + i.ToString().
            PadLeft(pad) + "]");
        }
        for (int j = 0; j < M[0].Length; ++j)
        {
          double v = M[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).
            PadLeft(wid));
        }
        Console.WriteLine("");
      }
    }

    // ------------------------------------------------------

    static void MatShow(double[][] M, int dec,
      int wid, int nRows)
    {
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < nRows; ++i)
      {
        for (int j = 0; j < M[0].Length; ++j)
        {
          double v = M[i][j];
          if (Math.Abs(v) < small) v = 0.0;
          Console.Write(v.ToString("F" + dec).
            PadLeft(wid));
        }
        Console.WriteLine("");
      }
      if (nRows < M.Length)
        Console.WriteLine(". . . ");
    }

    // ------------------------------------------------------

    static double[][] MatStandardize(double[][] data,
      out double[] means, out double[] stds)
    {
      // scikit style z-score biased normalization
      int rows = data.Length;
      int cols = data[0].Length;
      double[][] result = MatCreate(rows, cols);

      // compute means
      double[] mns = new double[cols];
      for (int j = 0; j < cols; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i)
          sum += data[i][j];
        mns[j] = sum / rows;
      } // j

      // compute std devs
      double[] sds = new double[cols];
      for (int j = 0; j < cols; ++j)
      {
        double sum = 0.0;
        for (int i = 0; i < rows; ++i)
          sum += (data[i][j] - mns[j]) *
            (data[i][j] - mns[j]);
        sds[j] = Math.Sqrt(sum / rows);  // biased
      } // j

      // normalize
      for (int j = 0; j < cols; ++j)
      {
        for (int i = 0; i < rows; ++i)
          result[i][j] = (data[i][j] - mns[j]) / sds[j];
      } // j

      means = mns;
      stds = sds;

      return result;
    }

    // ------------------------------------------------------

    static double[][] MatCreate(int rows, int cols)
    {
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
      return result;
    }

    // ------------------------------------------------------

    static double[][] CovarMatrix(double[][] data,
      bool rowVar)
    {
      // rowVar == true means each row is a variable
      // if false, each column is a variable

      double[][] source;
      if (rowVar == true)
        source = data;  // by ref
      else
        source = MatTranspose(data);

      int srcRows = source.Length;  // num features
      int srcCols = source[0].Length;  // not used

      double[][] result = MatCreate(srcRows, srcRows);

      for (int i = 0; i < result.Length; ++i)
      {
        for (int j = 0; j <= i; ++j)
        {
          result[i][j] = Covariance(source[i], source[j]);
          result[j][i] = result[i][j];
        }
      }

      return result;
    }

    // ------------------------------------------------------

    static double Covariance(double[] v1, double[] v2)
    {
      // compute means of v1 and v2
      int n = v1.Length;

      double sum1 = 0.0;
      for (int i = 0; i < n; ++i)
        sum1 += v1[i];
      double mean1 = sum1 / n;

      double sum2 = 0.0;
      for (int i = 0; i < n; ++i)
        sum2 += v2[i];
      double mean2 = sum2 / n;

      // compute covariance
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
        sum += (v1[i] - mean1) * (v2[i] - mean2);
      double result = sum / (n - 1);

      return result;
    }

    // ------------------------------------------------------

    static double[][] MatTranspose(double[][] M)
    {
      int nr = M.Length;
      int nc = M[0].Length;
      double[][] result = MatCreate(nc, nr);  // note
      for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nc; ++j)
          result[j][i] = M[i][j];
      return result;
    }

    // ------------------------------------------------------

    static void Eigen(double[][] A,
      out double[] eval, out double[][] evec)
    {
      // Jacobi algorithm based on GSL implementation
      // assumes A is square symmetric
      // OK because being applied to a covariance matrix
      int m = A.Length; int n = A[0].Length;
      // if m != n throw an exception
      int maxRot = 100 * m * m;  // heuristic
      double redSum = 0.0;

      eval = new double[m];
      evec = MatIdentity(m);

      for (int i = 0; i < maxRot; ++i)
      {
        double nrm = Normalize(A);

        if (nrm == 0.0)  // mildly risky
          break;

        for (int p = 0; p < n; ++p)
        {
          for (int q = p + 1; q < n; ++q)
          {
            double c; double s;
            redSum += Symschur2(A, p, q, out c, out s);
            // Compute A := J^T A J 
            Apply_Jacobi_L(A, p, q, c, s);
            Apply_Jacobi_R(A, p, q, c, s);
            // Compute V := V J 
            Apply_Jacobi_R(evec, p, q, c, s);
          }
        }
      }

      // nrot = i;
      for (int p = 0; p < n; ++p)
      {
        double ep = A[p][p];
        eval[p] = ep;
      }
    }

    // ------------------------------------------------------

    static double[][] MatIdentity(int n)
    {
      double[][] result = MatCreate(n, n);
      for (int i = 0; i < n; ++i)
        result[i][i] = 1.0;
      return result;
    }

    // ------------------------------------------------------

    static double Symschur2(double[][] A, int p, int q,
      out double c, out double s)
    {
      // Symmetric Schur decomposition 2x2 matrix
      double Apq = A[p][q];
      if (Apq != 0.0)
      {
        double App = A[p][p];
        double Aqq = A[q][q];
        double tau = (Aqq - App) / (2.0 * Apq);
        double t, c1;

        if (tau >= 0.0)
          t = 1.0 / (tau + Hypot(1.0, tau));
        else
          t = -1.0 / (-tau + Hypot(1.0, tau));

        c1 = 1.0 / Hypot(1.0, t);
        c = c1; s = t * c1;
      }
      else  // Apq == 0.0
      {
        c = 1.0; s = 0.0;
      }

      return Math.Abs(Apq);
    }

    // ------------------------------------------------------

    static double Hypot(double x, double y)
    {
      // wacky sqrt(x^2 + y^2)
      double xabs = Math.Abs(x);
      double yabs = Math.Abs(y);
      double min, max;

      if (xabs < yabs)
      {
        min = xabs; max = yabs;
      }
      else
      {
        min = yabs; max = xabs;
      }

      if (min == 0)
        return max;

      double u = min / max;
      return max * Math.Sqrt(1 + u * u);
    }

    // ------------------------------------------------------

    static void Apply_Jacobi_L(double[][] A, int p,
      int q, double c, double s)
    {
      int n = A[0].Length;

      // Apply rotation to matrix A,  A' = J^T A 
      for (int j = 0; j < n; ++j)
      {
        double Apj = A[p][j];
        double Aqj = A[q][j];
        A[p][j] = Apj * c - Aqj * s;
        A[q][j] = Apj * s + Aqj * c;
      }
    }

    // ------------------------------------------------------

    static void Apply_Jacobi_R(double[][] A, int p,
      int q, double c, double s)
    {
      int m = A.Length;

      // Apply rotation to matrix A,  A' = A J 
      for (int i = 0; i < m; ++i)
      {
        double Aip = A[i][p];
        double Aiq = A[i][q];
        A[i][p] = Aip * c - Aiq * s;
        A[i][q] = Aip * s + Aiq * c;
      }
    }

    // ------------------------------------------------------

    static double Normalize(double[][] A)
    {
      int m = A.Length; int n = A[0].Length;
      double scale = 0.0;
      double ssq = 1.0;

      for (int i = 0; i < m; ++i)
      {
        for (int j = 0; j < n; j++)
        {
          double Aij = A[i][j];

          // compute norm of off-diagonal elements
          if (i == j) continue;

          if (Aij != 0.0)
          {
            double ax = Math.Abs(Aij);

            if (scale < ax)
            {
              ssq =
                1.0 + ssq * (scale / ax) * (scale / ax);
              scale = ax;
            }
            else
            {
              ssq += (ax / scale) * (ax / scale);
            }
          }

        } // j
      } // i

      double sum = scale * Math.Sqrt(ssq);
      return sum;
    }

    // ------------------------------------------------------

    static int[] ArgSort(double[] vec)
    {
      int n = vec.Length;
      int[] idxs = new int[n];
      for (int i = 0; i < n; ++i)
        idxs[i] = i;
      Array.Sort(vec, idxs);  // sort idxs based on vec vals
      return idxs;
    }

    // ------------------------------------------------------

    static void VecShow(double[] vec, int dec, int wid)
    {
      double small = 1.0 / Math.Pow(10, dec);
      for (int i = 0; i < vec.Length; ++i)
      {
        double x = vec[i];
        if (Math.Abs(x) < small) x = 0.0;  // avoid "-0.0"
        Console.Write(x.ToString("F" + dec).PadLeft(wid));
      }
      Console.WriteLine("");
    }

    // ------------------------------------------------------

    static double[][] MatExtractCols(double[][] mat,
      int[] cols)
    {
      int srcRows = mat.Length;
      int srcCols = mat[0].Length;
      int tgtCols = cols.Length;

      double[][] result = MatCreate(srcRows, tgtCols);
      for (int i = 0; i < srcRows; ++i)
      {
        for (int j = 0; j < tgtCols; ++j)
        {
          int c = cols[j];
          result[i][j] = mat[i][c];
        }
      }
      return result;
    }

    // ------------------------------------------------------

    static double[][] MatProduct(double[][] matA,
      double[][] matB)
    {
      int aRows = matA.Length;
      int aCols = matA[0].Length;
      int bRows = matB.Length;
      int bCols = matB[0].Length;
      if (aCols != bRows)
        throw new Exception("Non-conformable matrices");

      double[][] result = MatCreate(aRows, bCols);

      for (int i = 0; i < aRows; ++i) // each row of A
        for (int j = 0; j < bCols; ++j) // each col of B
          for (int k = 0; k < aCols; ++k)
            result[i][j] += matA[i][k] * matB[k][j];

      return result;
    }

    // ------------------------------------------------------

    static double[] VarExplained(double[] eigenVals)
    {
      // assumes eigenVals are sorted large to small
      int n = eigenVals.Length;
      double[] result = new double[n];
      double sum = 0.0;
      for (int i = 0; i < n; ++i)
        sum += eigenVals[i];
      for (int i = 0; i < n; ++i)
      {
        double pctExplained = eigenVals[i] / sum;
        result[i] = pctExplained;
      }
      return result;
    }

  } // Program

} // ns
